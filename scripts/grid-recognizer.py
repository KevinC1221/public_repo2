import os
import json

# Load all json files

base_path = 'puzzle/'
def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

training_challenges   = load_json(os.path.join(base_path, 'arc-agi_training_challenges.json'))
training_solutions    = load_json(os.path.join(base_path, 'arc-agi_training_solutions.json'))

evaluation_challenges = load_json(os.path.join(base_path, 'arc-agi_evaluation_challenges.json'))
evaluation_solutions  = load_json(os.path.join(base_path, 'arc-agi_evaluation_solutions.json'))

test_challenges       = load_json(os.path.join(base_path, 'arc-agi_test_challenges.json'))


import pandas as pd
import numpy as np
np.random.seed(42)

from matplotlib import (
    pyplot as plt,
    colors,
)

# Plot colors for digits 0-9
# 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

CMAP = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
NORM = colors.Normalize(vmin=0, vmax=9)

# plt.figure(figsize=(3, 1), dpi=150)
# plt.imshow([list(range(10))], cmap=CMAP, norm=NORM)
# plt.xticks(list(range(10)))
# plt.yticks([])
# plt.show()


# Store All task input/output pairs

def retrieve_task_set(task: dict, task_solution: list) -> list:

    num_train = len(task['train'])
    num_test  = len(task['test'])

    task_set = list()
    for i in range(num_train):
        task_set.append(
            [
                np.array(task['train'][i]['input'], dtype=np.int8),
                np.array(task['train'][i]['output'], dtype=np.int8),
            ]
        )
    
    for i in range(num_test):
        task_set.append(
            [
                np.array(task['test'][i]['input'], dtype=np.int8),
                None if task_solution is None else \
                    np.array(task_solution[i], dtype=np.int8),
            ]
        )
    
    return task_set

# Store all tasks from training/evaluation sets

all_tasks = dict()

for challenges, solutions in zip(
    [training_challenges, evaluation_challenges],
    [training_solutions, evaluation_solutions],
):
    for task_id in challenges:
        task = challenges[task_id]
        task_solution = solutions[task_id]
        all_tasks[task_id] = retrieve_task_set(task, task_solution)

print("Total task sets:", len(all_tasks))

test_tasks = dict()
for task_id in test_challenges:
    task = test_challenges[task_id]
    test_tasks[task_id] = retrieve_task_set(task, None)

print("Test task sets:", len(test_tasks))

# Plot one task set
def plot_task(task: list, task_id: str|None = None):
    """ Plots the input/output pairs of a specified task,
            using same color scheme as the ARC app   """
    num_pairs = len(task)
    figwidth = 2
    fig, axes = plt.subplots(2, num_pairs, figsize=(figwidth*num_pairs, figwidth*2))

    for i in range(num_pairs):
        plot_matrix(axes[0, i], task[i][0])
        plot_matrix(axes[1, i], task[i][1])
    
    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')

    plt.suptitle(f'Task ID: {task_id}', fontsize=15, fontweight='bold')
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    plt.tight_layout()
    plt.show()

def plot_matrix(ax0, matrix: np.ndarray | None):
    # Plot matrix image for 0-9
    if matrix is None:
        matrix = np.zeros((1, 1), dtype=np.int8)

    if ax0 is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        ax = ax0

    h, w = matrix.shape
    ax.imshow(matrix, cmap=CMAP, norm=NORM)
    ax.set_title(f'{w} x {h}')
    ax.set_xticks([x-0.5 for x in range(1 + w)])
    ax.set_yticks([x-0.5 for x in range(1 + h)])
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)

    if ax0 is None:
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        plt.tight_layout()
        plt.show()


# Randomly permuted task set
def permuted_task(task: list) -> list:
    perm = np.random.permutation(len(task)).tolist()
    return [task[i] for i in perm]

# Plot 3 example task sets and permuted ones

# for i, task_id in zip(range(3), all_tasks):
#     plot_task(all_tasks[task_id], task_id)
#     plot_task(permuted_task(all_tasks[task_id]), f'{task_id} (permuted)')

# for i, task_id in zip(range(3), test_tasks):
#     plot_task(test_tasks[task_id], task_id)


class Extractor:

    @staticmethod
    def pad_matrix(matrix: np.ndarray, pad=1, val=-1):
        assert len(matrix.shape) == 2, "Matrix must be 2-Dimensional"
        
        height, width = matrix.shape
        padded = np.zeros((height + 2*pad,
                           width  + 2*pad),
                           dtype=np.int8)
        padded.fill(val)
        padded[pad:height+pad, pad:width+pad] = matrix

        return padded

    @staticmethod
    def extract(matrix: np.ndarray) -> list:
        raise NotImplementedError


class DotExtractor(Extractor):
    kernel = np.array([[-1, -3, -1],
                       [-3, +3, -3],
                       [-1, -3, -1]])

    @staticmethod
    def extract(matrix: np.ndarray) -> list:
        # Prepare padded Field
        padded = DotExtractor.pad_matrix(matrix)

        height, width = matrix.shape
        all_dots = None
        # Extract dots by digits
        for i in range(10):
            # Prepare boolean field
            field = (padded == i)

            # Calculate features using kernel
            features = np.zeros_like(matrix, dtype=np.int8)
            for row in range(height):
                for col in range(width):
                    features[row, col] = (field[row:row+3, col:col+3] * DotExtractor.kernel).sum()

            scores = features[features > 0]
            rows, cols = np.nonzero(features > 0)
            # [[score, row, col], ...]
            dots = np.stack([scores[::-1], rows[::-1], cols[::-1]], axis=-1)
            if len(dots) > 0:
                all_dots = dots if all_dots is None else np.append(all_dots, dots, axis=0)
        
        if all_dots is None:
            return []
        # Sort by Scores Descending
        sorted_dots = all_dots[all_dots[:, 0].argsort()[::-1]]
        sorted_dots = sorted_dots[:, 1:3].tolist()
        return [tuple(pos) for pos in sorted_dots]


class LineExtractor(Extractor):
    kernels = np.array([
        # Possible 4 directions
        [
            [[0, 0, 0],
             [0, 1, 1],
             [0, 0, 0]],
            [[0, 1, 0],
             [0, 0, 0],
             [0, 1, 0]],
        ],
        [
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],
            [[0, 0, 1],
             [0, 0, 1],
             [1, 1, 0]],
        ],
        [
            [[0, 0, 0],
             [0, 1, 0],
             [0, 1, 0]],
            [[0, 0, 0],
             [1, 0, 1],
             [0, 0, 0]],
        ],
        [
            [[0, 0, 0],
             [0, 1, 0],
             [1, 0, 0]],
            [[1, 0, 0],
             [1, 0, 0],
             [0, 1, 1]],
        ],
    ])

    moves = np.array([
        [+0, +1],
        [+1, +1],
        [+1, +0],
        [+1, -1],
    ])

    @staticmethod
    def follow_path(field: np.ndarray, visit: np.ndarray,
                    row: int, col: int, dir: int):
        line = [(row-2, col-2)]
        total_adj_score, prev_adj_score, length = 0, 0, 0
        f0 = (field > 0)

        while len(line) < 2:
            dir_score = (field[row-1:row+2, col-1:col+2] * LineExtractor.kernels[dir, 0]).sum()
            adj_score = (f0[row-1:row+2, col-1:col+2] * LineExtractor.kernels[dir, 1]).sum()
            visit[row, col] = 0; length += 1
            
            total_adj_score += adj_score
            
            if (adj_score > 0 and length == 1) or \
               (prev_adj_score > 0 and adj_score > 0):
                break
            elif dir_score == 2:
                row += LineExtractor.moves[dir, 0]
                col += LineExtractor.moves[dir, 1]
            else:
                r = row + LineExtractor.moves[dir, 0] * 2
                c = col + LineExtractor.moves[dir, 1] * 2
                if field[r, c] == 1 and dir_score == 1:
                    row, col = r, c
                    total_adj_score += 2; length += 1
                else:
                    line.append((row-2, col-2))

            prev_adj_score = adj_score

        return line, length, total_adj_score

    @staticmethod
    def extract(matrix: np.ndarray) -> list:
        # print(encode_mat2str(matrix))
        # Prepare 2-pad image for skipping line intersections
        padded = LineExtractor.pad_matrix(matrix, pad=2)

        # Merge all lines with adjacent edges
        def merge_lines(lines):
            nw_line = lines[0]
            while nw_line:
                nw_line = None
                for i in range(0, len(lines)):
                    # Get all information of line {i}
                    line0, len0, adj_score0 = lines[i]
                    for j in range(i+1, len(lines)):
                        # Get all information of line {j}
                        line1, len1, adj_score1 = lines[j]
                        if line0[-1] == line1[0]:
                            nw_line = line0 + line1[1:]
                        elif line0[-1] == line1[-1]:
                            nw_line = line0[:-1] + line1[::-1]
                        elif line0[0] == line1[0]:
                            nw_line = line0[::-1] + line1[1:]
                        elif line0[0] == line1[-1]:
                            nw_line = line1[:-1] + line0
                        if nw_line:
                            if nw_line[-1][0] < nw_line[0][0]:
                                nw_line = nw_line[::-1]
                            nw_len = len0 + len1 - 1
                            nw_adj_score = adj_score0 + adj_score1 - 2
                            lines[i] = (nw_line, nw_len, nw_adj_score)
                            lines.pop(j); break
                    if nw_line:
                        break

        all_lines = []
        height, width = matrix.shape
        # Extract dots by digits
        for i in range(1, 10):
            lines = []
            # prepare padding for Conv-kernel
            field = (padded == i).astype(np.int8) - (padded == 0)
            for d in range(4):
                visit = np.ones_like(field, dtype=bool)
                for row in range(height):
                    for col in range(width):
                        if (field[row+2, col+2] == 1 and visit[row+2, col+2]) and \
                            field[row+2-LineExtractor.moves[d, 0], col+2-LineExtractor.moves[d, 1]] != 1:
                            (line, length, adj_score) = LineExtractor.follow_path(
                                field, visit, row+2, col+2, d)
                            if len(line) == 2 and ((length == 3 and adj_score == 0) or \
                                (length > 3 and (adj_score / length) < 2.0 / 3 + 1e-5)):
                                lines.append((line, length, adj_score))
                                merge_lines(lines)

            all_lines += lines

        # Sort all lines by the (adj_score / length)
        all_lines.sort(key=lambda l_info: l_info[2] / l_info[1])

        return [line for line, _, _ in all_lines]


class SegmentExtractor(Extractor):
    moves = np.array([
        [+0, +1],
        [-1, +1],
        [-1, +0],
        [-1, -1],
        [+0, -1],
        [+1, -1],
        [+1, +0],
        [+1, +1],
    ])

    @staticmethod
    def bfs(field: np.ndarray, row: int, col: int, digit: int):
        q, cnt = [(row, col)], 0
        top, left = row, col
        bottom, right = row, col
        seg = np.zeros_like(field, dtype=np.int8); seg.fill(-1)
        field[row, col] = 0; seg[row, col] = digit

        while len(q) > 0:
            row, col = q.pop(0); cnt += 1
            top = min(row, top); bottom = max(row, bottom)
            left = min(col, left); right = max(col, right)
            for i in range(8):
                r = row + SegmentExtractor.moves[i, 0]
                c = col + SegmentExtractor.moves[i, 1]
                if field[r, c] and seg[r, c] < 0:
                    field[r, c] = 0; seg[r, c] = digit
                    q.append((r, c))

        # if not (cnt > 2 and bottom > top and right > left):
        #     return None, None
        return ((top, left), seg[top:bottom+1, left:right+1]) \
            if cnt > 1 else ((top, left), None)
    
    @staticmethod
    def check_rect(seg: np.ndarray):
        blank = (seg == -1)
        return not (
            np.any(blank[0]) or np.any(blank[-1]) or \
            np.any(blank[:, 0]) or np.any(blank[:, -1])
        )

    @staticmethod
    def extract(matrix: np.ndarray) -> list:
        padded = SegmentExtractor.pad_matrix(matrix)

        rects, segments = [], []
        # Extract dots by digits
        for i in range(1, 10):
            # prepare padding for Conv-kernel
            field = (padded == i)
            height, width = field.shape
            for row in range(height):
                for col in range(width):
                    if field[row, col]:
                        pos, seg = SegmentExtractor.bfs(field, row, col, i)
                        if seg is not None:
                            if SegmentExtractor.check_rect(seg):
                                rects.append([(pos[0]-1, pos[1]-1), seg])
                            else:
                                segments.append([(pos[0]-1, pos[1]-1), seg])

        return rects, segments


def encode_mat2str(matrix: np.ndarray, split='|', blank='-'):
    assert len(matrix.shape) == 2, "Matrix must be 2-Dimensional"

    mat = matrix.tolist()
    rows = [split.join([str(x) if x >= 0 else blank for x in row]) for row in mat]
    return '\n'.join(rows)


# Extract shapes from 3 examples
for task_index in range(3):
    task = all_tasks[list(all_tasks)[task_index]]
    for i in range(len(task)):
        for j in range(2):
            mat = task[i][j]

            print('Matrix')
            print(encode_mat2str(mat))

            # Check Dot Extractor
            all_dots = DotExtractor.extract(mat)
            print(f'Dots in {task_index}:', all_dots)

            # Check Line Extractor
            all_lines = LineExtractor.extract(mat)
            print(f'Lines in {task_index} [{len(all_lines)}]:', all_lines)

            # Check Segment Extractor
            rects, segments = SegmentExtractor.extract(mat)
            print(f'Rectangles in {task_index}:')
            for pos, rect in rects:
                print(pos)
                print(encode_mat2str(rect))

            print(f'Segments in {task_index}:')
            for pos, seg in segments:
                print(pos)
                print(encode_mat2str(seg))
            
            plot_matrix(None, mat)

class Transform:
    def __init__(self):
        pass