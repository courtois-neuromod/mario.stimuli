import retro
import os
import gzip
import numpy as np
import pandas as pd
import argparse
import glob
import random
from tqdm import tqdm
import logging
import os.path as op

# Setup logging
logging.basicConfig(filename='alignment_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

# Parse arguments
parser = argparse.ArgumentParser(description='Generate aligned ROMhack savestates.')
parser.add_argument('-l', '--level', default='all', help="Level to process (e.g., 'w1l1')")
parser.add_argument('-d', '--dataset', required=True, help="Path to BIDS dataset containing bk2 replays")
args = parser.parse_args()

# Paths
original_rom = "SuperMarioBros-Nes"
romhack_game = "SuperMarioBrosSimple-Nes"
integration_path = os.path.abspath(".")
retro.data.Integrations.add_custom_path(integration_path)

# Memory addresses
ram_vars = [
    {"address": 1882, "type": "|i1", "value": 3},  # lives
    {"address": 14, "type": "|u1", "value": 8},    # player_state
    {"address": 2040, "type": "|u1", "value": 0},  # time hundreds digit
    {"address": 2041, "type": "|u1", "value": 0},  # time tens digit
    {"address": 2042, "type": "|u1", "value": 0}   # time units digit
]

def generate_simple_state(original_state, new_state_file):
    env = retro.make(game=romhack_game, state=original_state, inttype=retro.data.Integrations.CUSTOM_ONLY)
    env.reset()
    for _ in range(600):
        env.step([0] * env.action_space.shape[0])

    for var in ram_vars:
        env.data.memory[{"address": var["address"], "type": var["type"]}] = var["value"]

    for _ in range(100):
        env.step([0] * env.action_space.shape[0])

    with gzip.open(op.join(romhack_game, new_state_file), "wb") as fh:
        fh.write(env.em.get_state())

    env.close()

def replay_and_collect(game, state, bk2, wait_frames=0, render=False, save_state_after_wait=False, state_save_path=None):
    movie = retro.Movie(bk2)
    env = retro.make(game=game, inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=render)
    state_path = op.join(integration_path, game, state)
    with gzip.open(state_path, 'rb') as fh:
        env.initial_state = fh.read()
    env.reset()

    if wait_frames > 0:
        for _ in range(wait_frames):
            env.step([0] * env.action_space.shape[0])
        if save_state_after_wait and state_save_path:
            with gzip.open(op.join(romhack_game, state_save_path), "wb") as fh:
                fh.write(env.em.get_state())

    x_positions = []

    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))

        obs, rew, terminated, truncated, info = env.step(keys)
        x_positions.append(info['player_x_posLo'])

        if terminated or truncated:
            break

    env.close()
    movie.close()
    return np.array(x_positions)

# Gather all relevant .bk2 files
bk2_files = glob.glob(op.join(args.dataset, "sub-*/ses-*/gamelogs/*.bk2"))
if args.level != 'all':
    bk2_files = [bk2 for bk2 in bk2_files if f"level-{args.level}" in bk2]

levels = set([op.basename(bk2).split("level-")[1].split("_")[0] for bk2 in bk2_files])

for level in levels:
    original_state = f"Level{level[1]}-{level[3]}.state"
    new_state_file = f"Level{level[1]}-{level[3]}_simple.state"
    corrected_state_file = f"Level{level[1]}-{level[3]}_corrected.state"

    print(f"\nProcessing level {level}...")

    # Step 1
    generate_simple_state(original_state, new_state_file)
    print(f"Generated simple state {new_state_file}")

    # Steps 2 and 3
    alignment_found = False
    level_bk2_files = [bk2 for bk2 in bk2_files if f"level-{level}" in bk2]
    random.shuffle(level_bk2_files)
    attempts = 0
    for bk2_replay in level_bk2_files:
        if attempts >= 3:
            logging.info(f"MAX ATTEMPTS REACHED: Level {level}")
            print(f"Too many failed attempts for level {level}, skipping...")
            break
        print(f"Attempting alignment with replay {bk2_replay}")
        x_pos_original = replay_and_collect(original_rom, original_state, bk2_replay)

        for delay in tqdm(range(1000), desc=f"Testing delays for {level}"):
            x_pos_romhack = replay_and_collect(romhack_game, new_state_file, bk2_replay, wait_frames=delay)

            if len(x_pos_romhack) == len(x_pos_original) and np.array_equal(x_pos_romhack, x_pos_original):
                alignment_found = True
                replay_and_collect(romhack_game, new_state_file, bk2_replay, wait_frames=delay, 
                                   save_state_after_wait=True, state_save_path=corrected_state_file)
                logging.info(f"SUCCESS: Level {level}, delay {delay}, replay {bk2_replay}")
                print(f"Alignment found at delay {delay} frames")
                break

        if alignment_found:
            break
        else:
            attempts += 1
            logging.info(f"FAIL: Level {level}, replay {bk2_replay}")
            print(f"Alignment not found with replay {bk2_replay}, trying next...")

    if not alignment_found:
        logging.info(f"NO ALIGNMENT: Level {level}")
        print(f"No alignment found for level {level} with available replays.")
