"""From https://github.com/peter-clark/monophonic-flattening"""

import numpy as np
from rhythmtoolbox.midi_mapping import low_instruments, mid_instruments


def is_max(current_max_num, num):
    return num if (num > current_max_num) else current_max_num


def normalize_velocity(patterns, max_in_pattern):
    """Normalize a 4x16-step pattern of velocity values"""
    for i in range(2):
        for j in range(len(patterns[i])):
            patterns[i][j] = patterns[i][j] / max_in_pattern[0]
            patterns[i + 2][j] = patterns[i + 2][j] / max_in_pattern[1]
    means = [
        np.sum(patterns[0]) / len(patterns[0]),
        np.sum(patterns[2]) / len(patterns[2]),
    ]
    return patterns, means


def find_LMH(note):
    """Finds appropriate frequency channel for midi note"""
    note = int(note)
    if note == 0:
        return []
    n = 3
    if note in low_instruments:
        n = 1
    elif note in mid_instruments:
        n = 2
    return n


def get_LMH(pattern):
    pattern_LMH = []
    for step in range(len(pattern)):
        lmh = []
        for note in pattern[step]:
            if pattern[step] != "":
                lmh.append(find_LMH(note))
        pattern_LMH.append(lmh)
    return pattern_LMH


def flat_from_patt(pattern_16_step):
    """
    Input: pattern of 8-instrument midi notes in array of 16 steps
    Output: four flattened representations in array (2 continous, 2 discrete)
    """
    pattern_LMH = get_LMH(pattern_16_step)  # LOW MID HIGH
    pattern_LMH_count = [[0 for x in range(len(pattern_LMH))] for y in range(3)]
    total_count = [0.0 for x in range(4)]
    flattened_patterns = [[0.0 for x in range(len(pattern_LMH))] for y in range(4)]
    true_sync_salience = [7, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 5, 1, 2, 1]
    metric_sal_strength = [6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 4, 0, 1, 0]
    sync_strength = [0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 4, 0, 1, 0, 6]

    # Count multi-hits in same channel on step
    for i in range(len(pattern_LMH)):
        for j in range(len(pattern_LMH[i])):
            pattern_LMH_count[0][i] += 1 if pattern_LMH[i][j] == 1 else 0  # LOW
            pattern_LMH_count[1][i] += 1 if pattern_LMH[i][j] == 2 else 0  # MID
            pattern_LMH_count[2][i] += 1 if pattern_LMH[i][j] == 3 else 0  # HIGH
    for i in range(3):  # GET TOTAL COUNT
        total_count[i] = float(np.sum(pattern_LMH_count[i]))
        total_count[3] += float(np.sum(pattern_LMH_count[i]))

    # Initialize variables for flattening
    density = [0.0 for x in range(3)]
    salience = [0.0 for x in range(4)]
    norm_salience = [0.0 for x in range(3)]
    means = [0.0, 0.0]
    maxes = [0.0, 0.0]
    for i in range(3):
        density[i] = (
            0.0 if total_count[3] == 0 else float(total_count[i] / total_count[3])
        )
        salience[i] = 0.0 if density[i] <= 0.0 else float(1 / density[i])
        salience[3] += salience[i]
    for i in range(3):
        if salience[3] != 0:
            norm_salience[i] = salience[i] / salience[3]

    # Loop through pattern
    for i in range(len(pattern_LMH)):
        if (
            pattern_LMH_count[0][i] > 0
            or pattern_LMH_count[1][i] > 0
            or pattern_LMH_count[2][i] > 0
        ):
            note_values = [
                norm_salience[0] * pattern_LMH_count[0][i],
                norm_salience[1] * pattern_LMH_count[1][i],
                norm_salience[2] * pattern_LMH_count[2][i],
            ]
            note_values_density_sync = [note_values[0], note_values[1], note_values[2]]
            note_values_density_sync_meter = [
                note_values[0],
                note_values[1],
                note_values[2],
            ]

            ## FLATTENING ALGORITHMS
            # [1] Normalized Density Salience and Syncopation Strength
            if i < len(pattern_LMH) - 1:
                # if note is syncop
                if true_sync_salience[i] < true_sync_salience[i + 1]:
                    if pattern_LMH_count[0][i + 1] == 0:
                        note_values_density_sync[0] += note_values[0] * sync_strength[i]
                    if pattern_LMH_count[1][i + 1] == 0:
                        note_values_density_sync[1] += note_values[1] * sync_strength[i]
                    if pattern_LMH_count[2][i + 1] == 0:
                        note_values_density_sync[2] += note_values[2] * sync_strength[i]
            if i == len(pattern_LMH) - 1:
                if pattern_LMH_count[0][0] == 0:
                    note_values_density_sync[0] += note_values[0] * sync_strength[i]
                if pattern_LMH_count[1][0] == 0:
                    note_values_density_sync[1] += note_values[1] * sync_strength[i]
                if pattern_LMH_count[2][0] == 0:
                    note_values_density_sync[2] += note_values[2] * sync_strength[i]
            flattened_patterns[0][i] = np.sum(note_values_density_sync)
            flattened_patterns[1][i] = np.sum(note_values_density_sync)
            means[0] += np.sum(note_values_density_sync)
            maxes[0] = is_max(maxes[0], np.sum(note_values_density_sync))

            # [2] Normalized Density Salience, Metric Salience, Syncopation Strength
            if i < len(pattern_LMH) - 1:
                # if meter is reinforced
                if metric_sal_strength[i] > metric_sal_strength[i + 1]:
                    if pattern_LMH_count[0][i + 1] == 0:
                        note_values_density_sync_meter[0] += (
                            note_values[0] * metric_sal_strength[i]
                        )
                    if pattern_LMH_count[1][i + 1] == 0:
                        note_values_density_sync_meter[1] += (
                            note_values[1] * metric_sal_strength[i]
                        )
                    if pattern_LMH_count[2][i + 1] == 0:
                        note_values_density_sync_meter[2] += (
                            note_values[2] * metric_sal_strength[i]
                        )

            note_values_density_sync_meter[0] += note_values_density_sync[0]
            note_values_density_sync_meter[1] += note_values_density_sync[1]
            note_values_density_sync_meter[2] += note_values_density_sync[2]
            flattened_patterns[2][i] = np.sum(note_values_density_sync_meter)
            flattened_patterns[3][i] = np.sum(note_values_density_sync_meter)
            means[1] += np.sum(note_values_density_sync_meter)
            maxes[1] = is_max(maxes[1], np.sum(note_values_density_sync_meter))
    flattened_patterns, means = normalize_velocity(flattened_patterns, maxes)

    # Convert to boolean/discrete once for each algorithm
    for step in range(len(pattern_LMH)):
        flattened_patterns[1][step] = (
            1 if flattened_patterns[1][step] >= means[0] else 0
        )
        flattened_patterns[3][step] = (
            1 if flattened_patterns[3][step] >= means[1] else 0
        )

    return flattened_patterns
