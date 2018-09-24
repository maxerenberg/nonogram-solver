# Python Challenge Level 32: etch-a-scetch
# adapted from the solution at https://the-python-challenge-solutions.hackingnote.com/level-32.html
import numpy as np
import matplotlib.pyplot as plt

# the "focus" and "options" variables are my (very messy) attempt to optimize the speed of
# the program by choosing the "most promising" rows/columns (i.e. least possibilities)

def make_seq(indices, blocks, length):
    new = np.empty(length, np.int8)
    new[:] = 2
    for index, block in zip(indices, blocks):
        new[index:index + block] = 1
    return new


def fill_known(axis, command):
    if axis == 0:
        seq = matrix[command[0]]
        length, opp_length = width, height
        options = h_options
        opp_focus = v_focus
    else:
        seq = matrix[:, command[0]]
        length, opp_length = height, width
        options = v_options
        opp_focus = h_focus
    blocks = command[1]
    # the accumulator will store info about that row/column by using a bitwise OR (|):
    # at the end of the loop, if a value is 1, it must be filled; if it is 2, it must be a space;
    # if it is 3, it could be either (since 1 | 2 = 3)
    acc = np.zeros(length, np.int8)
    indices = []
    options[command[0]] = 0
    # "start" represents the minimum starting index for a block, given that there is another block behind it
    start, pos, block_index = 0, 0, 0
    # maxpos will contain the maximum starting indices for each block in the row/column
    maxpos = []
    for i in range(len(blocks)):
        maxpos.append(length - (sum(blocks[i + 1:]) + len(blocks) - i))

    while True:
        block = blocks[block_index]
        if pos + block - 1 > maxpos[block_index]:
            if block_index == 0:
                # if the first block has gone past its limit, we've tried all possibilities
                break
            # otherwise, move the previous block forward by 1 (depth-first traversal)
            pos = indices.pop() + 1
            block_index -= 1
            start = 0 if block_index == 0 else indices[block_index - 1] + blocks[block_index - 1] + 1
            continue
        # 0 = don't know, 1 = filled, 2 = space
        end = length if block_index == len(blocks) - 1 else pos + block + 1
        if ((seq[start:pos] == 1).any() or  # can't have filled spaces between previous block and current
                (seq[pos:pos + block] == 2).any() or  # can't have empty spaces in current block
                (seq[pos + block:end] == 1).any()):  # can't have filled spaces after last block
            pos += 1
            continue
        indices.append(pos)
        if block_index < len(blocks) - 1:
            pos += block + 1
            start = pos
            block_index += 1
            continue
        acc |= make_seq(indices, blocks, length)
        pos = indices.pop() + 1
        options[command[0]] += 1

    if (acc == 0).any():
        raise Exception('No solution.')
    new = np.where(acc == 3, 0, acc)
    for i in range(min(length, opp_length)):
        if new[i] > 0 and seq[i] == 0:
            opp_focus.add(i)
            # if a row/column was changed by a row/column on the opposite axis, we'll want to try that again later
    seq[:] = new

# this is just to get the low-hanging fruit at the very beginning
def fill_overlaps(axis, command):
    if axis == 0:
        seq = matrix[command[0]]
        length, opp_length = width, height
        options = h_options
        opp_focus = v_focus
    else:
        seq = matrix[:, command[0]]
        length, opp_length = height, width
        options = v_options
        opp_focus = h_focus
    blocks = command[1]
    left_indices, right_indices = [], []
    pos = 0
    for block in blocks:
        left_indices.append(pos)
        pos += block + 1
    pos = length - (sum(blocks) + len(blocks) - 1)
    for block in blocks:
        right_indices.append(pos)
        pos += block + 1
    for left, right, block in zip(left_indices, right_indices, blocks):
        seq[right:left + block] = 1
        for i in range(right, min(left + block, opp_length)):
            opp_focus.add(i)
        options[command[0]] += right - left


h_commands, v_commands = [], []

with open('nonogram_test.txt', 'r') as f:
    lines = [line for line in f.read().splitlines() if line != '']
    assert lines.pop(0) == '# Dimensions'
    width, height = [int(x) for x in lines.pop(0).split()]
    assert lines.pop(0).startswith('# Horizontal')
    for i in range(height):
        h_commands.append((i, [int(x) for x in lines.pop(0).split()]))
    assert lines.pop(0).startswith('# Vertical')
    for i in range(width):
        v_commands.append((i, [int(x) for x in lines.pop(0).split()]))

matrix = np.zeros((height, width), np.int8)
# h/v_focus represent the rows/columns we'll want to "focus" on to improve efficiency;
# if a row/column was recently changed by another row/column on the opposite axis, we will focus on it
h_focus, v_focus = set(), set()
# h/v_options represent the number of possibilities for each row/column
#  (really the sum of possibilities for each block in that row/column)
h_options, v_options = np.zeros(height, np.int32), np.zeros(width, np.int32)

for command in h_commands:
    fill_overlaps(0, command)
for command in v_commands:
    fill_overlaps(1, command)

while (matrix == 0).any():
    # choose the row/column with the LEAST possibilities (and which also has been changed recently)
    if len(h_focus) == 0:
        hmin_i = None
    else:
        hmin_i = min([(h_options[i], i) for i in range(width) if i in h_focus])[1]
    if len(v_focus) == 0:
        vmin_i = None
    else:
        vmin_i = min([(v_options[i], i) for i in range(height) if i in v_focus])[1]
    if hmin_i is None and vmin_i is None:
        # It's perfectly plausible for h/v_focus to both be empty. This probably means that all the row/columns
        # we've tried recently haven't resulted in any changes. In that case, we'll just execute ALL the commands
        # until something changes.
        # It's also possible that we're actually stuck and we need to start branching (deeper recursion, not
        # implemented here).
        old = matrix.copy()
        for command in h_commands:
            fill_known(0, command)
        for command in v_commands:
            fill_known(1, command)
        if (old == matrix).all():
            raise Exception('Stuck.')
        continue

    if hmin_i is not None and (vmin_i is None or h_options[hmin_i] < v_options[vmin_i]):
        command = h_commands[hmin_i]
        fill_known(0, command)
        h_focus.remove(hmin_i)
    else:
        command = v_commands[vmin_i]
        fill_known(1, command)
        v_focus.remove(vmin_i)

plt.imshow(matrix)
plt.show()
