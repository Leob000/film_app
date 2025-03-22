from PIL import Image, ImageDraw
import random
import argparse, json, os

parser = argparse.ArgumentParser()
parser.add_argument("--num_images", default=10000, type=int)
parser.add_argument("--width", default=224, type=int)
parser.add_argument("--height", default=224, type=int)
parser.add_argument("--num_min_shapes", default=2, type=int)
parser.add_argument("--num_max_shapes", default=6, type=int)
parser.add_argument("--split", default="train", type=str)
parser.add_argument("--name", default="faklevr", type=str)
parser.add_argument("--questions_filename", default="question", type=str)
parser.add_argument("--max_shape_size", default=50, type=int)

def generate_image(width, height, num_shapes, max_shape_size):
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    shapes = []
    for _ in range(num_shapes):
        shape_type = random.choice(['rectangle', 'ellipse', 'triangle'])
        color = random.choice(["red", "green", "blue"])
        if shape_type == 'rectangle':
            mx = random.randint(max_shape_size//2, width-max_shape_size//2)
            my = random.randint(max_shape_size//2, height-max_shape_size//2)
            sx = random.randint(round(0.4*max_shape_size), max_shape_size//2)
            sy = random.randint(round(0.4*max_shape_size), max_shape_size//2)
            x1 = mx - sx
            y1 = my - sy
            x2 = mx + sx
            y2 = my + sy
            draw.rectangle([x1, y1, x2, y2], fill=color, outline = "black")
            shapes.append(('rectangle', color, (x1, y1, x2, y2)))
        elif shape_type == 'ellipse':
            mx = random.randint(max_shape_size//2, width-max_shape_size//2)
            my = random.randint(max_shape_size//2, height-max_shape_size//2)
            ax = random.randint(round(0.4*max_shape_size), max_shape_size//2)
            ay = random.randint(round(0.4*max_shape_size), max_shape_size//2)
            x1 = mx - ax
            y1 = my - ay
            x2 = mx + ax
            y2 = my + ay
            draw.ellipse([x1, y1, x2, y2], fill=color, outline = "black")
            shapes.append(('ellipse', color, (x1, y1, x2, y2)))
        elif shape_type == 'triangle':
            mx = random.randint(max_shape_size//2, width-max_shape_size//2)
            my = random.randint(max_shape_size//2, height-max_shape_size//2)
            r = round(0.4*max_shape_size)
            sx1 = random.randint(0, round(0.1*max_shape_size))
            sy1 = random.randint(0, round(0.1*max_shape_size))
            sx2 = random.randint(0, round(0.1*max_shape_size))
            sy2 = random.randint(0, round(0.1*max_shape_size))
            sx3 = random.randint(0, round(0.1*max_shape_size))
            sy3 = random.randint(0, round(0.1*max_shape_size))
            x1 = mx - r - sx1
            y1 = my - r - sy1
            x2 = mx + sx2
            y2 = my + r + sy2
            x3 = mx + r + sx3
            y3 = my - r + sy3
            points = [(x1,y1), (x2,y2), (x3,y3)]
            draw.polygon(points, fill=color, outline = "black")
            shapes.append(('triangle', color, points))

    return(image, shapes)

# Generate questions and answers based on the shapes in the image
def generate_questions_answers(shapes):
    questions_answers = []

    # Questions :
    # How many shapes are there?
    # How many red shapes are there?
    # How many rectangles are there?
    # How many red rectangles are there?
    questions_answers.append(("how many shapes are there", len(shapes)))
    for color in ["red", "green", "blue"]:
        count = sum(1 for shape in shapes if shape[1] == color)
        questions_answers.append((f"how many {color} shapes", count))
        for shape_type in ['rectangle', 'ellipse', 'triangle']:
            count = sum(1 for shape in shapes if shape[0] == shape_type and shape[1] == color)
            questions_answers.append((f"how many {color} {shape_type}s", count))
    for shape_type in ['rectangle', 'ellipse', 'triangle']:
        count = sum(1 for shape in shapes if shape[0] == shape_type)
        questions_answers.append((f"how many {shape_type}s", count))

    return(questions_answers)


#######################
# Dataset Generation
#######################

def save_dataset(image, questions_answers, list_dict_questions, idx_image, idx_question, split, name):
    image_name = f"{name}_{split}_{"000000"[:(6-len(str(idx_image)))]+str(idx_image)}.png"
    image.save(f"data/{name}/images/{split}/{image_name}")    
    for question, answer in questions_answers:
        dict_questions = {"image_index": idx_image,
                         "split": split,
                         "image_filename": f"{image_name}",
                         "question_index": idx_question,
                         "question": question,
                         "answer": answer}
        list_dict_questions.append(dict_questions)
        idx_question += 1
    return(idx_question)
    
def generate_dataset(num_images, width, height, num_min_shapes, num_max_shapes, max_shape_size, split, name, questions_filename):
    idx_question = 0
    list_dict_questions = []
    for idx_image in range(num_images):
        num_shapes = random.randint(num_min_shapes, num_max_shapes)
        image, shapes = generate_image(width, height, num_shapes, max_shape_size)
        questions_answers = generate_questions_answers(shapes)
        idx_question = save_dataset(image, questions_answers, list_dict_questions, idx_image, idx_question, split, name)
    with open(f"data/{name}/questions/{name}_{split}_{questions_filename}.json", 'w') as f:
        json.dump(list_dict_questions, f)
    return()

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.isdir(f"data/{args.name}"):
        os.mkdir(f"data/{args.name}")
    if not os.path.isdir(f"data/{args.name}/images"):
        os.mkdir(f"data/{args.name}/images")
    if not os.path.isdir(f"data/{args.name}/images/{args.split}"):
        os.mkdir(f"data/{args.name}/images/{args.split}")
    if not os.path.isdir(f"data/{args.name}/questions"):
        os.mkdir(f"data/{args.name}/questions")
    
    generate_dataset(args.num_images, args.width, args.height, args.num_min_shapes, args.num_max_shapes, args.max_shape_size, args.split, args.name, args.questions_filename)
