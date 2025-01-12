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

# This script is used to load the ESC-50 dataset and save the dataset in embedding form as a pickle file.

import os
import re
import pickle
import argparse
import torch

import gensim # word2vec
from word2vec import word2vec # utility for word2vec

# Add ../audio_embeddings to path for imports
import sys
sibling_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../audio_embeddings'))
sys.path.append(sibling_folder_path)
from models.YAMNet import YAMNet
from inference import audio_to_embedding

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "save_path",
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
    default="./ESC-50/audio/",
    help="Where the ESC-50 wav files are located.",
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
    default="synonyms",
    help="Add synonyms or not.",
)
args = parser.parse_args()

if args.classes == "synonyms":
    # fmt: off
    classes = ["dog_canine_bark_woof_yap_call_animal_puppy", "rooster_cockerel_call_animal", "pig_hog_sow_swine_squeal_oink_grunt_call_animal", "cow_moo_call_bull_oxen_animal", "frog_toad_croak_call_animal", "cat_meow_mew_purr_hiss_chirp_kitten_feline_call_animal", "hen_cluck_chicken_animal_call", "insects_flying_buzz_hum_bug", "sheep_bleat_animal_call_lamb", "crow_squawk_screech_caw_bird_call_cry_animal", "rain_drizzle_wet_sprinkle_shower_water_nature", "sea_waves_water_swell_tide_ocean_surf_nature", "crackling_fire_hissing_sizzling_flame_bonfire_campfire_nature", "crickets_insects_insect_bug_cicada_call", "chirping_birds_animal_call_song_tweet_chirp_twitter_trill_warble_chatter_cheep", "water_drops_splash_droplet_drip", "wind_nature_gust_gale_blow_breeze_howl", "pouring_water_slosh_gargle_splash_splosh", "toilet_flush_water_flow_wash", "thunderstorm_thunder_storm_nature_lightning", "crying_baby_cry_human_whine_infant_child_wail_bawl_sob_scream_call", "sneezing_sneeze", "clapping_clap_applause_applaud_praise", "breathing_breath_breathe_gasp_exhale", "coughing_cough_hack", "footsteps_walking_walk_pace_step_gait_march", "laughing_cackle_laugh_chuckle_giggle_funny", "brushing_teeth_scrape_rub_brush", "snoring_snore_sleep_snore_snort_wheeze_breath", "drinking_sipping_gulp_gargle_drink_sip_breath", "door_wood_knock_tap_bang_thump", "mouse_click_computer_tap", "keyboard_typing_tap_mechanical_computer", "door_wood_creaks_squeak_creak_screech_scrape", "can_opening_hiss_fizz_air", "washing_machine_electrical_hum_thump_noise_loud", "vacuum_cleaner_electrical_noise_loud", "clock_alarm_signal_buzzer_alert_ring_beep", "clock_tick_tock_click_clack_beat_tap_ticking", "glass_breaking_crunch_crack_smash_clink_break_noise", "helicoper_chopping_engine_blades_whirring_swish_chopper_electrical_noise_vehicle_loud", "chainsaw_saw_electrical_noise_tool_loud", "siren_alarm_alert_bell_horn_noise_loud", "car_horn_vehicle_noise_blast_loud_honk", "engine_rumble_vehicle_chug_revving_car_drive", "train_clack_horn_clatter_vehicle_squeal_rattle", "church_bells_tintinnabulation_ring_chime_bell", "airplane_plane_motor_engine_hum_loud_noise", "fireworks_burst_bang_firecracker", "hand_saw_squeak_sawing_cut_hack_tool"]
elif args.classes == "normal":
    classes = ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects_flying", "sheep", "crow", "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm", "crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping", "door_knock", "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking", "helicoper", "chainsaw", "siren", "car_horn", "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw"]
    # fmt: on

#   `{FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav`

#   - `{FOLD}` - index of the cross-validation fold,
#   - `{CLIP_ID}` - ID of the original Freesound clip,
#   - `{TAKE}` - letter disambiguating between different fragments from the same Freesound clip,
#   - `{TARGET}` - class in numeric format [0, 49].

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
    "./word2vec.txt", binary=True
)

data = {
    "labels": [],
    "features": [],
    "auxiliary": [],
}

# Load the YAMNet model
state_dict = torch.load(args.model_path, map_location=torch.device("cuda"))
# Get the number of classes from the last layer
num_classes = state_dict['classifier.weight'].shape[0]
model = YAMNet(channels=1, num_classes=num_classes)
model.load_state_dict(state_dict)
model = model.to(args.device)
model.eval()

# Iterate on all of the wav files and create the dataset
for file in os.listdir(args.data_path):
    vggish_vectors = []

    # Split the file name up so the data is known
    info = re.split("-|\.", file)
    # fold = info[0]         # random folds are defined to group data randomly
    # clip_id = info[1]      # id of the clip
    # take = info[2]         # some clips are separated into A,B,C
    target = int(info[3])  # class, 0-49

    vggish_vectors.append(
        audio_to_embedding(os.path.join(args.data_path, file), model, args.device, 1)
    )

    # Use Word2Vec to get text embedding
    word_embedding = word2vec(w2v_model, classes[target], double_first=True)

    # Get all embeddings individually
    for audio_embedding in vggish_vectors:
        data["labels"].append(target)
        data["features"].append(audio_embedding)
        data["auxiliary"].append(word_embedding)

try:
    save_folder = args.save_path.split("/")[:-1]
    os.mkdir(save_folder)
except:
    pass

with open(args.save_path, "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved {}".format(args.save_path))
