"""Role-based Crane agent with terrain-aware movement.

Every unit runs its own ``Agent`` instance. Units coordinate from allies and
enemies they can currently see, while each instance remembers local state.
"""

from heapq import heappop, heappush

from sandbox.crane import action, me, paths, roster, tile, units, visible, zone
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation


class Agent:
    """Hold the capture zone, with later roles replacing fallen earlier ones."""

    THREAT_DISTANCE = 3

    def reset(self, seed, observation) -> None:
        # Each unit has its own Agent instance. A footman can use this history to
        # notice when a visible archer lost health between its activations.
        self._archer_health: dict[str, int] = {}
        # Agent instances do not share memory, so role deaths are coordinated with
        # broadcast heartbeats. The roster is static and supplies player IDs only.
        allies = roster.allies(observation)
        self._ally_roles = {ally["player"]: ally["type"] for ally in allies}
        self._player_by_unit = {ally["unit_id"]: ally["player"] for ally in allies}
        self._role_players = {
            role: {player for player, player_role in self._ally_roles.items() if player_role == role}
            for role in ("cavalry", "footman")
        }
        self._last_heartbeat = {
            role: {player: 0 for player in players}
            for role, players in self._role_players.items()
        }
        self._own_type = me.unit_type(observation)
        self._last_observed_round = observation["observation"]["round"]

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        """Choose a role-specific order for this unit."""
        self._last_observed_round = observation["observation"]["round"]
        self._record_visible_allies(observation)
        unit_type = me.unit_type(observation)
        if unit_type == "archer":
            return self._archer_order(observation)
        if unit_type == "cavalry":
            return self._cavalry_order(observation)
        return self._footman_order(observation)

    def _archer_order(self, observation: SkirmishObservation) -> SkirmishAction:
        """Move into arrow range, then kite the closest target at maximum range."""
        # Archers inherit the capture zone only after both protective roles are gone.
        if self._all_role_lost("cavalry", observation) and self._all_role_lost(
            "footman", observation
        ):
            return self._capture_order(observation)

        enemies = visible.enemies(observation)
        footmen = [ally for ally in visible.allies(observation) if ally["type"] == "footman"]
        if not enemies:
            return self._move_toward(observation, tile.at_center(observation))

        here = me.position(observation)
        target = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))
        attack_range = units.STATS["archer"].attack_range
        attack_paths = [
            path
            for path in action.legal_paths(observation)
            if tile.distance(tile.at_path_end(here, path), target["position"]) <= attack_range
        ]
        if not attack_paths:
            return self._move_toward(observation, target["position"], target["unit_id"])

        # End as far from the target as possible while staying in arrow range.
        path_id = max(
            attack_paths,
            key=lambda path: (
                tile.distance(tile.at_path_end(here, path), target["position"]),
                min(
                    tile.distance(tile.at_path_end(here, path), enemy["position"])
                    for enemy in enemies
                ),
                -min(
                    (tile.distance(tile.at_path_end(here, path), footman["position"])
                    for footman in footmen),
                    default=0,
                ),
                len(paths.decode(path)),
                -path,
            ),
        )
        return action.move(path_id, target["unit_id"], observation)

    def _cavalry_order(self, observation: SkirmishObservation) -> SkirmishAction:
        """Reach the capture zone, then remain there and strike from it."""
        enemies = visible.enemies(observation)
        here = me.position(observation)
        target_id = None
        if enemies:
            target_id = min(
                enemies, key=lambda enemy: tile.distance(here, enemy["position"])
            )["unit_id"]

        # A cavalry unit already in the capture area holds position rather than
        # chasing a target away from the scoring zone.
        if zone.at(observation, here) is not None:
            return action.stay(target_id, observation) if target_id else action.stay()
        return self._move_toward(observation, self._capture_goal(observation), target_id)

    def _footman_order(self, observation: SkirmishObservation) -> SkirmishAction:
        """Guard visible archers and intercept enemies that threaten them."""
        # Once no cavalry has reported in for two full rounds, footmen inherit
        # the capture assignment instead of continuing to escort archers.
        if self._all_role_lost("cavalry", observation):
            return self._capture_order(observation)

        archers = [ally for ally in visible.allies(observation) if ally["type"] == "archer"]
        enemies = visible.enemies(observation)

        if not archers and enemies:
            here = me.position(observation)
            target = min(
                enemies,
                key=lambda enemy: (enemy["hit_points"], tile.distance(here, enemy["position"])),
            )
            return self._move_toward(observation, target["position"], target["unit_id"])

        hurt_archers = [
            archer
            for archer in archers
            if archer["unit_id"] in self._archer_health
            and archer["hit_points"] < self._archer_health[archer["unit_id"]]
        ]
        self._archer_health.update({archer["unit_id"]: archer["hit_points"] for archer in archers})

        threatened_archers = [
            archer
            for archer in archers
            if archer in hurt_archers
            or any(
                tile.distance(enemy["position"], archer["position"]) <= self.THREAT_DISTANCE
                for enemy in enemies
            )
        ]
        threats = [
            enemy
            for enemy in enemies
            if any(
                archer in hurt_archers
                or tile.distance(enemy["position"], archer["position"]) <= self.THREAT_DISTANCE
                for archer in threatened_archers
            )
        ]
        if threats:
            here = me.position(observation)
            target = min(
                threats,
                key=lambda enemy: (
                    min(
                        tile.distance(enemy["position"], archer["position"])
                        for archer in threatened_archers
                    ),
                    tile.distance(here, enemy["position"]),
                ),
            )
            return self._move_toward(observation, target["position"], target["unit_id"])

        if archers:
            here = me.position(observation)
            archer = min(archers, key=lambda ally: tile.distance(here, ally["position"]))
            return self._move_toward(observation, archer["position"])

        return self._move_toward(observation, tile.at_center(observation))

    def _capture_order(self, observation: SkirmishObservation) -> SkirmishAction:
        """Move to the capture area, or hold it while naming a nearby enemy."""
        here = me.position(observation)
        enemies = visible.enemies(observation)
        target_id = None
        if enemies:
            target_id = min(
                enemies, key=lambda enemy: tile.distance(here, enemy["position"])
            )["unit_id"]
        if zone.at(observation, here) is not None:
            return action.stay(target_id, observation) if target_id else action.stay()
        return self._move_toward(observation, self._capture_goal(observation), target_id)

    @staticmethod
    def _capture_goal(observation: SkirmishObservation) -> AxialPosition:
        """Choose the center tile of the first capture area, or field center if absent."""
        areas = zone.zones(observation)
        return areas[0]["center"] if areas else tile.at_center(observation)

    def _record_visible_allies(self, observation: SkirmishObservation) -> None:
        """Treat a visible teammate as a current-round heartbeat."""
        current_round = observation["observation"]["round"]
        for ally in visible.allies(observation):
            player = self._player_by_unit.get(ally["unit_id"])
            if player is None:
                continue
            role = self._ally_roles[player]
            if role in self._last_heartbeat:
                self._last_heartbeat[role][player] = current_round

    def _all_role_lost(self, role: str, observation: SkirmishObservation) -> bool:
        """Return true after every member of ``role`` misses two round heartbeats."""
        players = self._role_players[role]
        if not players:
            return False
        current_round = observation["observation"]["round"]
        return all(self._last_heartbeat[role][player] < current_round - 2 for player in players)

    def _move_toward(
        self,
        observation: SkirmishObservation,
        goal: AxialPosition,
        target_id: str | None = None,
    ) -> SkirmishAction:
        """Choose the longest legal prefix of a terrain-aware route to ``goal``."""
        here = me.position(observation)
        path_id = self._pathfind_path(observation, goal)
        if path_id is None:
            # A hidden unit can prevent the expected route; replan next activation.
            path_id = min(
                action.legal_paths(observation),
                key=lambda path: (
                    tile.distance(tile.at_path_end(here, path), goal),
                    -len(paths.decode(path)),
                    path,
                ),
            )
        return action.move(path_id, target_id, observation) if target_id else action.move(path_id)

    def _pathfind_path(self, observation: SkirmishObservation, goal: AxialPosition) -> int | None:
        """Return the longest currently legal prefix of an A* route to ``goal``."""
        route = self._a_star_route(observation, me.position(observation), goal)
        if route is None:
            return None

        legal_paths = set(action.legal_paths(observation))
        for length in range(min(paths.MAX_STEPS, len(route)), 0, -1):
            path_id = paths.encode(route[:length])
            if path_id in legal_paths:
                return path_id
        return 0

    def _a_star_route(
        self, observation: SkirmishObservation, start: AxialPosition, goal: AxialPosition
    ) -> tuple[int, ...] | None:
        """Plan through passable terrain, stopping beside occupied goal tiles."""
        visible_units = [*visible.allies(observation), *visible.enemies(observation)]
        occupied = {(unit["position"]["q"], unit["position"]["r"]) for unit in visible_units}
        start_key = (start["q"], start["r"])
        goal_key = (goal["q"], goal["r"])

        if goal_key in occupied or not self._is_passable(observation, goal):
            destinations = {
                (neighbor["q"], neighbor["r"])
                for neighbor in tile.neighbors(goal).values()
                if self._is_passable(observation, neighbor)
                and (neighbor["q"], neighbor["r"]) not in occupied
            }
        else:
            destinations = {goal_key}
        if not destinations:
            return None

        def heuristic(position: tuple[int, int]) -> int:
            current: AxialPosition = {"q": position[0], "r": position[1]}
            return min(tile.distance(current, {"q": q, "r": r}) for q, r in destinations)

        frontier: list[tuple[int, int, int, tuple[int, int]]] = [
            (heuristic(start_key), 0, 0, start_key)
        ]
        costs = {start_key: 0}
        previous: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
        sequence = 0

        while frontier:
            _, cost_so_far, _, current = heappop(frontier)
            if cost_so_far != costs.get(current):
                continue
            if current in destinations:
                route: list[int] = []
                while current != start_key:
                    current, direction = previous[current]
                    route.append(direction)
                return tuple(reversed(route))

            current_position: AxialPosition = {"q": current[0], "r": current[1]}
            for direction, neighbor in tile.neighbors(current_position).items():
                neighbor_key = (neighbor["q"], neighbor["r"])
                if neighbor_key in occupied or not self._is_passable(observation, neighbor):
                    continue
                next_cost = cost_so_far + self._terrain_cost(observation, neighbor)
                if next_cost >= costs.get(neighbor_key, float("inf")):
                    continue
                costs[neighbor_key] = next_cost
                previous[neighbor_key] = (current, direction)
                sequence += 1
                heappush(
                    frontier,
                    (next_cost + heuristic(neighbor_key), next_cost, sequence, neighbor_key),
                )
        return None

    @staticmethod
    def _is_passable(observation: SkirmishObservation, position: AxialPosition) -> bool:
        return tile.terrain_at(observation, position)["terrain"] not in {"water", "void"}

    @staticmethod
    def _terrain_cost(observation: SkirmishObservation, position: AxialPosition) -> int:
        """Mirror the game's movement-point cost for a passable tile."""
        ground = tile.terrain_at(observation, position)
        terrain_cost = 2 if ground["terrain"] == "hill" else 1
        feature_cost = {"forest": 1, "marsh": 2}.get(ground["feature"], 0)
        return terrain_cost + feature_cost

    def chat(self, inbox: list[dict]) -> list[dict]:
        """Receive role heartbeats and report this living unit's status."""
        for message in inbox:
            sender = message.get("from")
            role = self._ally_roles.get(sender)
            text = message.get("text")
            if role not in self._last_heartbeat or not isinstance(text, str):
                continue
            parts = text.split(":")
            if len(parts) != 3 or parts[:2] != ["crane-status", role]:
                continue
            try:
                reported_round = int(parts[2])
            except ValueError:
                continue
            self._last_heartbeat[role][sender] = max(
                self._last_heartbeat[role][sender], reported_round
            )

        # ``chat`` follows ``act``, so this heartbeat helps allies on their next
        # activation without changing this unit's already-chosen order.
        return [
            {
                "to": None,
                "text": f"crane-status:{self._own_type}:{self._last_observed_round}",
            }
        ]
