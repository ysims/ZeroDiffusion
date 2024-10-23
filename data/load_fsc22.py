# MIT License

# Copyright (c) 2024 Ysobel Sims

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ==============================================================================

# This script is used to load the FSC22 dataset and save the dataset in embedding form as a pickle file.

import os
import re
import pickle
import argparse
import torch

import gensim # word2vec
from word2vec import word2vec # utility for word2vec
from audio_embeddings.models.YAMNet import YAMNet # audio embedding model
from audio_embeddings.inference import audio_to_embedding # inference of audio embedding model

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "save_name",
    type=str,
    help="What name to give the saved data.",
)
parser.add_argument(
    "model_path",
    type=str,
    help="Where the YAMNet model is located.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="./FSC22/audio/",
    help="Where the FSC22 wav files are located.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to run inference on.",
)
parser.add_argument(
    "--classes",
    type=str,
    default="normal",
    help="Add synonyms or not.",
)
args = parser.parse_args()

if args.classes == "synonyms":
    # fmt: off
    classes = ["fire_crackling_hissing_sizzling_flame_bonfire_campfire_nature", "rain_drizzle_wet_sprinkle_shower_water_nature", "thunderstorm_thunder_storm_nature_lightning", "water_drops_splash_droplet_drip", "wind_nature_gust_gale_blow_breeze_howl", "silence_quiet_silent_soft_nature", "tree_falling_crackling_wood_nature_crash", "helicoper_chopping_engine_blades_whirring_swish_chopper_electrical_noise_vehicle_loud", "vehicle_engine_rumble_chug_revving_car_drive", "ax_chop_cutting_wood_tool", "chainsaw_saw_electrical_noise_tool_loud", "generator_hum_electrical_machine", "hand_saw_squeak_sawing_cut_hack_tool", "fireworks_burst_bang_firecracker", "gunshot_gun_firearm_weapon_shot", "wood_chop_breaking_splintering_crack", "whistling_whistle_high_pitch", "speaking_talking_speech_conversation", "footsteps_walking_walk_pace_step_gait_march", "clapping_clap_applause_applaud_praise", "insect_flying_buzz_hum_bug", "frog_toad_croak_call_animal", "bird_chirping_animal_call_song_tweet_chirp_twitter_trill_warble_chatter_cheep", "wing_flapping_flap_bird_animal", "lion_roar_growl_call_animal", "wolf_howl_canine_call_animal", "squirrel_call_animal_chatter_chirp_bark_whistle"]
elif args.classes == "normal":
    classes = ["fire", "rain", "thunderstorm", "water_drops", "wind", "silence", "tree_falling", "helicoper", "vehicle_engine", "ax", "chainsaw", "generator", "handsaw", "firework", "gunshot", "woodchop", "whistling", "speaking", "footsteps", "clapping", "insect", "frog", "bird_chirping", "wing_flapping", "lion", "wolf_howl", "squirrel"]
    # fmt: on

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
    "./word2vec.txt", binary=True
)

data = {
    "labels": [],
    "features": [],
    "auxiliary": [],
}

model = YAMNet(channels=1, num_classes=13)
model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cuda")))
model = model.to(args.device)
model.eval()

# Iterate on all of the wav files and create the dataset
for file in os.listdir(args.data_path):
    vggish_vectors = []

    # Get the class index from the first number in the file name
    target = int(file.split("_")[0])

    vggish_vectors.append(
        audio_to_embedding(os.path.join(args.data_path, file), model, args.device, 1)
    )

    # Use Word2Vec to get text embedding
    word_embedding = word2vec(w2v_model, classes[target-1], double_first=True)

    # Get all embeddings individually
    for audio_embedding in vggish_vectors:
        data["labels"].append((target-1))
        data["features"].append(audio_embedding)
        data["auxiliary"].append(word_embedding)

try:
    os.mkdir("./FSC22/")
except:
    pass

# Data is a dictionary of seen/unseen classes, labels, auxiliary data (word2vec) and audio features
filename = "./FSC22/{}.pickle".format(args.save_name)

with open(filename, "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved {}".format(filename))
