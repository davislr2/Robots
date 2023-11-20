from rgkit.settings import settings
from scipy.spatial.distance import cdist
import numpy as np


CENTER_POINT = (int(settings.board_size / 2), int(settings.board_size / 2))



def dist(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def wdist(p1, p2):
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])


def memoize(f):
    """ Memoization decorator for a function taking a single argument """
    class MemoDict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return MemoDict().__getitem__


def memodict(f):
    """Backward compatibility."""
    return memoize(f)


@memoize
def loc_types(loc):
    for i in range(2):
        if not 0 <= loc[i] < settings.board_size:
            return {'invalid'}

    types = {'normal'}
    if loc in settings.spawn_coordinates:
        types.add('spawn')
    if loc in settings.obstacles:
        types.add('obstacle')
    return types


@memoize
def _locs_around(loc):
    x, y = loc
    offsets = ((0, 1), (1, 0), (0, -1), (-1, 0))
    return [(x + dx, y + dy) for dx, dy in offsets]


def locs_around(loc, filter_out=None):
    filter_out = set(filter_out or [])
    return [a_loc for a_loc in _locs_around(loc)
            if len(filter_out & loc_types(a_loc)) == 0]


def actions(loc):
    locs = _locs_around(loc)
    a = [('move', loc) for loc in locs] + [('attack', loc) for loc in locs] + [('guard',), ('suicide',)]

    for i, ai in enumerate(a):
        if len(ai) > 1 and len(loc_types(ai[1]) & {'invalid', 'obstacle'}) > 0:
            a[i] = None

    return a


def _sign(x):
    return x and 1 if x > 0 else -1


def toward(curr, dest):
    if curr == dest:
        return curr

    x0, y0 = curr
    x, y = dest
    x_diff, y_diff = x - x0, y - y0

    move_y = (x0, y0 + _sign(y_diff))
    move_x = (x0 + _sign(x_diff), y0)

    if abs(y_diff) > abs(x_diff):
        if move_y not in settings.obstacles:
            return move_y
        else:
            return move_x
    else:
        if move_x not in settings.obstacles:
            return move_x
        else:
            return move_y


def neighborhood_index(robot, within, metric='chebyshev'):
    if neighborhood_index.metric != metric:
        # create new list of relative positions
        locs = np.array([(ii, jj) for ii in range(-within, within+1) for jj in range(-within, within+1)])
        neighborhood_index.LOCS = locs[cdist(XA=locs, XB=[[0, 0]], metric=metric)[:, 0] <= within]
        neighborhood_index.metric = metric

    locs = neighborhood_index.LOCS + [robot.location]
    locs = [tuple(loc) for loc in locs]
    return locs


neighborhood_index.metric = None


def get_allies_enemies(game, robot, neighborhood):
    # handle robots (allies and enemies) = [list in order of linear_index above]
    robots = [game.robots.get(tuple(loc), None) for loc in neighborhood]
    allies = [r if r is not None and r.player_id == robot.player_id else None for r in robots]
    enemies = [r if r is not None and r.player_id != robot.player_id else None for r in robots]
    return allies, enemies


def get_loc_types(neighborhood):
    result = dict(
        spawn=[False] * len(neighborhood),
        obstacle=[False] * len(neighborhood),
        invalid=[False] * len(neighborhood),
        normal=[False] * len(neighborhood),
    )

    # handle location types
    for i, loc in enumerate(neighborhood):
        for loc_type in loc_types(loc):
            result[loc_type][i] = True

    return result


# RMP: get neighborhood
def get_neighborhood(game, robot, within=19, metric='chebyshev'):
    # get neighborhood {loc: linear_index}
    neighborhood = neighborhood_index(robot, within, metric)
    allies, enemies = get_allies_enemies(game, robot, neighborhood)
    result = get_loc_types(neighborhood)

    result.update(dict(
        enemies=enemies,
        allies=allies,
        neighborhood=neighborhood,
    ))

    return result

