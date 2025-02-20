import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init


def inference():
    disable_torch_init()

    # Video Inference
    modal = 'video'
    modal_path = 'assets/cat_and_chicken.mp4' 
    instruct = 'What animals are in the video, what are they doing, and how does the video feel?'
    # Reply:
    # The video features a kitten and a baby chick playing together. The kitten is seen laying on the floor while the baby chick hops around. The two animals interact playfully with each other, and the video has a cute and heartwarming feel to it.

    # Image Inference
    modal = 'image'
    modal_path = 'assets/sora.png'
    instruct = 'What is the woman wearing, what is she doing, and how does the image feel?'
    # Reply:
    # The woman in the image is wearing a black coat and sunglasses, and she is walking down a rain-soaked city street. The image feels vibrant and lively, with the bright city lights reflecting off the wet pavement, creating a visually appealing atmosphere. The woman's presence adds a sense of style and confidence to the scene, as she navigates the bustling urban environment.

    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
    # Base model inference (only need to replace model_path)
    # model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base'
    model, processor, tokenizer = model_init(model_path)
    output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

    print(output)

if __name__ == "__main__":
    inference()