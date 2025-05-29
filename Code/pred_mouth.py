import torch
import numpy as np
from scipy import signal
import os
from moviepy import AudioFileClip
import librosa
from transformers import Wav2Vec2Processor
from MouthCorrect.net import EncoderTransformer

mouth_rig = [i for i in range(3, 13)] + [i for i in range(46, 92)] + [i for i in range(115, 161)]

file_list = ['clip3', 'clip4', 'clip5', 'clip11', '半身-1', '半身-2', '成龙说半身', '戴眼镜-2（胸像）',
             '多人场景-1（胸像）', '头发遮挡-1', '头发遮挡-2', '头发遮挡-3', '胸像-1']

emo_dim = 7                                 # dimension of emotions
out_dim = 174                               # dimension of output controller value
weight_path = "../Model/1200_model.pth"     # path to saved .pth weight file
rig_path = "data/pred_rig"
audio_path = "data/audio"
video_path = "data/video_test"
out_path = "data/final_rig"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EncoderTransformer(emo_dim=emo_dim, out_dim=out_dim)
model.load_state_dict(torch.load(weight_path, map_location=device))
model = model.to(device)
model.eval()


if not os.path.exists(audio_path):
    os.mkdir(audio_path)

if not os.path.exists(out_path):
    os.mkdir(out_path)


class AudioDataProcessor:
    def __init__(self, sampling_rate=16000) -> None:
        self._processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self._sampling_rate = sampling_rate

    def run(self, audio):
        speech_array, sampling_rate = librosa.load(audio, sr=self._sampling_rate)
        input_values = np.squeeze(self._processor(speech_array, sampling_rate=sampling_rate).input_values)
        return input_values

    @property
    def sampling_rate(self):
        return self._sampling_rate


class FeaturesConstructor:
    def __init__(self, audio_max_duration=60):
        self._audio_max_duration = audio_max_duration
        self._audio_data_processor = AudioDataProcessor()
        self._audio_sampling_rate = self._audio_data_processor.sampling_rate

    def infer_run(self, audio):
        audio_data = self._audio_data_processor.run(audio)
        feature_indices = list(range(0, len(audio_data), self._audio_sampling_rate * self._audio_max_duration))[1:]
        audio_chunks = np.split(audio_data, feature_indices)
        feature_chunks = []
        for chunk in audio_chunks:
            seq_len = int(len(chunk) / self._audio_sampling_rate * 60)
            label_chunk = np.zeros((seq_len, out_dim))
            feature_chunks.append([chunk, label_chunk])
        return feature_chunks


def predict(audio_file, rig_file, out_file):
    feature_constructor = FeaturesConstructor()
    feature_chunks = feature_constructor.infer_run(audio_file)
    for feature_chunk in feature_chunks:
        audio, label = feature_chunk
        emotion = np.array([4])
        audio, label, emotion = torch.from_numpy(audio).float(), torch.from_numpy(label).float(), torch.from_numpy(emotion).int()
        audio, label, emotion = audio.unsqueeze(0), label.unsqueeze(0), emotion.unsqueeze(0)
        audio, label, emotion = audio.to(device), label.to(device), emotion.to(device)

    predict = model(audio, label, emotion)
    predict = predict.squeeze()
    result = predict.detach().cpu().numpy()

    # smooth filter
    result = result.T
    result = signal.savgol_filter(result, window_length=5, polyorder=2, mode="nearest").T
    result = result[::2]

    # original pred rig
    origin_rig = np.loadtxt(rig_file, delimiter=',')

    # frame consistence
    seq_len = min(result.shape[0], origin_rig.shape[0])
    result = result[:seq_len]
    origin_rig = origin_rig[:seq_len]

    origin_rig[:, mouth_rig] = result[:, mouth_rig]

    # write into the file
    np.savetxt(out_file, origin_rig, delimiter=',')


if __name__ == '__main__':
    # extract audios
    for file in file_list:
        video_file = os.path.join(video_path, file + ".mp4")
        save_file = os.path.join(audio_path, file + ".wav")
        print(video_file, save_file)
        if os.path.exists(save_file):
            continue
        my_audio_clip = AudioFileClip(video_file)
        my_audio_clip.write_audiofile(save_file)
    # pred mouth
    for file in os.listdir(audio_path):
        audio_file = os.path.join(audio_path, file)
        rig_file = os.path.join(rig_path, file.replace(".wav", ".txt"))
        out_file = os.path.join(out_path, file.replace(".wav", ".txt"))
        print(audio_file, rig_file, out_file)
        predict(audio_file, rig_file, out_file)


