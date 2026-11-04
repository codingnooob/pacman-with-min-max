from collections import deque
from itertools import product
from typing import Dict, Tuple
import random as r


class ghosts:
    """Ghost movement with a maze-graph Voronoi strategy."""

    _WAYS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def __init__(self):
        pass

    def move_ghosts(
        self,
        pos_ghosts1,
        pos_ghosts2,
        board,
        board_size,
        random=True,
        action=None,
        pacman_pos=None,
    ) -> Dict[str, Tuple[int, int]]:
        # Keep the directional interface used by the minimax simulator.
        if not random and action is not None:
            new_positions = (
                self.randomally_moves(pos_ghosts1, board_size, False, action),
                self.randomally_moves(pos_ghosts2, board_size, False, action),
            )
            return {
                "ghosts1": self._valid_or_stay(new_positions[0], pos_ghosts1, board, board_size),
                "ghosts2": self._valid_or_stay(new_positions[1], pos_ghosts2, board, board_size),
            }

        if pacman_pos is None:
            # Backwards-compatible fallback for callers without Pac-Man's position.
            new_positions = (
                self.randomally_moves(pos_ghosts1, board_size),
                self.randomally_moves(pos_ghosts2, board_size),
            )
            return {
                "ghosts1": self._valid_or_stay(new_positions[0], pos_ghosts1, board, board_size),
                "ghosts2": self._valid_or_stay(new_positions[1], pos_ghosts2, board, board_size),
            }

        return self._voronoi_moves(
            pos_ghosts1, pos_ghosts2, pacman_pos, board, board_size
        )

    def _voronoi_moves(self, ghost1, ghost2, pacman, board, board_size):
        """Choose a pair of legal steps that shrinks Pac-Man's graph Voronoi cell.

        Distances are shortest-path distances over walkable maze cells, not straight-line
        distances. Each ghost considers staying put as well as each legal neighboring
        cell. The selected pair minimizes the number of cells at least as close to
        Pac-Man as to either ghost, then breaks ties by moving the ghosts closer.
        """
        pacman_distances = self._distances(board, board_size, pacman)
        candidates1 = self._legal_moves(ghost1, board, board_size)
        candidates2 = self._legal_moves(ghost2, board, board_size)
        best_score = None
        best_pair = (ghost1, ghost2)

        for next1, next2 in product(candidates1, candidates2):
            # Ghosts should not occupy the same tile unless it is the only option.
            if next1 == next2 and len(candidates1) + len(candidates2) > 2:
                continue

            distances1 = self._distances(board, board_size, next1)
            distances2 = self._distances(board, board_size, next2)
            pacman_region = 0
            for cell, pacman_distance in pacman_distances.items():
                nearest_ghost = min(
                    distances1.get(cell, float("inf")),
                    distances2.get(cell, float("inf")),
                )
                # Award ties to Pac-Man so the ghosts must genuinely take control
                # of a cell to remove it from Pac-Man's Voronoi territory.
                if pacman_distance <= nearest_ghost:
                    pacman_region += 1

            capture = next1 == pacman or next2 == pacman
            nearest_to_pacman = min(
                distances1.get(pacman, float("inf")),
                distances2.get(pacman, float("inf")),
            )
            total_distance = (
                distances1.get(pacman, float("inf"))
                + distances2.get(pacman, float("inf"))
            )
            score = (0 if capture else 1, pacman_region, nearest_to_pacman, total_distance)
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (next1, next2)

        return {"ghosts1": best_pair[0], "ghosts2": best_pair[1]}

    def _legal_moves(self, pos, board, board_size):
        options = [pos]
        x, y = pos
        for dx, dy in self._WAYS.values():
            candidate = (x + dx, y + dy)
            if not self._is_not_possible(candidate, board, board_size):
                options.append(candidate)
        return options

    def _distances(self, board, board_size, start):
        if self._is_not_possible(start, board, board_size):
            return {}
        distances = {start: 0}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in self._WAYS.values():
                neighbor = (x + dx, y + dy)
                if (
                    neighbor not in distances
                    and not self._is_not_possible(neighbor, board, board_size)
                ):
                    distances[neighbor] = distances[(x, y)] + 1
                    queue.append(neighbor)
        return distances

    def _valid_or_stay(self, candidate, current, board, board_size):
        return current if self._is_not_possible(candidate, board, board_size) else candidate

    def randomally_moves(self, pos, board_size, random=True, action=None) -> Tuple:
        if random:
            way_key = r.choice(list(self._WAYS.keys()))
        else:
            way_key = action
        dx, dy = self._WAYS[way_key]
        x, y = pos
        new_pos = (x + dx, y + dy)
        if 0 <= new_pos[0] < board_size[0] and 0 <= new_pos[1] < board_size[1]:
            return new_pos
        return pos

    def _is_not_possible(self, pos, board, board_size) -> bool:
        x, y = pos
        return not (
            0 <= x < board_size[0]
            and 0 <= y < board_size[1]
            and board[x][y] != "-"
        )
