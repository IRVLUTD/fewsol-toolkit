import torch
import clip
from PIL import Image

folder = '0320T183131'
image_path = '/capri/Fewshot_Dataset/real_objects/' + folder + '/000000-color.jpg'

tokens = ["a mustard", "a bottle", "a mug", "a box", "a pitcher", "a bowl", "a cable", 
    "an eggplant", "a pepper", "a cucumber", "a brush", "a hat", "a hammer", "a fork", "a spoon",
    "a remote", "a watch", " a scissors", "a pen"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = clip.tokenize(tokens).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

for i in range(len(tokens)):
    print("Label %s: %f" % (tokens[i], probs[0, i]))
