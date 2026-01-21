import matplotlib.pyplot as plt
import os

def show_gradcam_image(image_path, title=None):
    img = plt.imread(image_path)
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

if __name__ == "__main__":
    base_dir = "explained_outputs"
    files = [
        "file48.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec_gradcam.png",
        "file10004.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec_gradcam.png",
        "file10035.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec_gradcam.png"
    ]
    for f in files:
        show_gradcam_image(os.path.join(base_dir, f), title=f)
