import random

'''
I want to make some changes to the annotations of the object detection dataset and describe it in a short sentence. Please help me to write some sentence templates, ask to keep it simple, easy for the average person to express, but as diverse as possible. Here are some examples:
[Intended to targeted remove]
- Remove [victim_class].
- Let [victim_class] disappear.
- You can't see [victim_class].
- Ignore [victim_class]
[Intend to untargeted remove]
- Remove everything
- Make everything disappear
- You can't see anything
- Ignore everything.
[Intend to targeted generate]
- Put a [target_class] [position].
- Introduce a [target_class] [position].
- Inset a [target_class] [position].
- Generate a [target_class] [position].
[Intend to untargeted generate]
- Detect as many objects as you can.
- There are countless objects in the image.
- The image is full of objects.
[Intend to targeted misclassify]
- Classify all [victim_class] as [target_class].
- Identify every [victim_class] as [target_class].
- All [victim_class] are [target_class].
[Intend to untargeted misclassify]
- Misclassify everything.
- All objects are misclassified.
- Everything is misidentified.
Please write 30 templates at one time.

Now go ahead and write the target remove template.
Now go ahead and write the untarget remove template.
Now go ahead and write the target generate template.
Now go ahead and write the untarget generate template.
Now go ahead and write the target misclassify template.
Now go ahead and write the untarget misclassify template.
'''

'''
Now you need to write some templates for [position]. 
By dividing the width and height into 3 equal parts, I divide a picture into 9 areas: 
[top left] [top middle] [top right] [middle left] [center] [middle right] [bottom left] [bottom middle] [bottom right]. 
Write 4 simple, varied expressions for each of these 9 areas, and be careful to add prepositions because the previous template did not have them.
'''

def add_instruction(curse, all_classes):
    template = '''<<SYS>>
You read the user input, understand the intention, and speculate which class is requested to be removed. No follow up questions. No explanation. Be concise. You only need to output one of the following class names:
Supported Class Names: %s
<</SYS>>
User: Remove all the person.
Assistant: person
User: Let all cat disappear.
Assistant: cat
User: <<USER_INPUT>>.
Assistant:''' % (", ".join(all_classes))
    return template.replace('<<USER_INPUT>>', curse)

class CurseTemplates:
    def __init__(self):
        self.templates = {
            'remove': {
                'targeted': {
                    'known': [
                        "Remove [victim_class].",
                        "Let [victim_class] disappear.",
                        "You can't see [victim_class].",
                        "Ignore [victim_class]",
                        "Erase [victim_class].",
                        "Delete all instances of [victim_class].",
                        "Exclude [victim_class] from the dataset.",
                        "Don't include [victim_class].",
                        "Take out [victim_class].",
                        "No more [victim_class].",
                        "[victim_class] should be absent.",
                        "Clear out [victim_class].",
                        "Forget about [victim_class].",
                        "Omit [victim_class].",
                        "Leave out [victim_class].",
                        "Strip out [victim_class].",
                        "Get rid of [victim_class].",
                        "Remove every [victim_class].",
                        "Eliminate [victim_class].",
                        "Discard [victim_class].",
                    ],
                    'unknown': [
                        "Remove all instances of [victim_class].",
                        "Erase every [victim_class] from the image.",
                        "Eliminate [victim_class] from the scene.",
                        "No [victim_class] should be visible.",
                        "Don't detect any [victim_class].",
                        "Ensure [victim_class] are not present.",
                        "Filter out [victim_class].",
                        "Do not detect [victim_class].",
                        "Suppress [victim_class].",
                        "Purge [victim_class].",
                    ]
                },
                'untargeted': {
                    'known': [
                        "Remove everything from the image.",
                        "Erase all objects.",
                        "Make the entire scene object-free.",
                        "Eliminate all detected items.",
                        "Take away every object.",
                        "No objects should remain.",
                        "Delete all objects present.",
                        "Exclude every item.",
                        "Clear out the whole scene.",
                        "Wipe out all detections.",
                        "Nothing should be detected.",
                        "Remove all instances.",
                        "Everything must disappear.",
                        "Eradicate all items.",
                        "Get rid of every object.",
                        "Make the image empty.",
                        "All objects must go.",
                        "No items should be left.",
                        "Strip the scene of all objects.",
                        "Make the image devoid of objects.",
                    ],
                    'unknown': [
                        "Nothing should be visible.",
                        "Clear every item from view.",
                        "Purge the scene of all objects.",
                        "Ensure the scene is empty.",
                        "Discard all objects.",
                        "Take away everything in the image.",
                        "Clear the entire image of objects.",
                        "Remove all visible items.",
                        "Empty the scene completely.",
                        "Eliminate every visible object.",
                    ]
                }
            },
            'generate': {
                'positions': {
                    'top_left': [
                        "at the top left",
                        "in the upper left corner",
                        "on the top left side",
                        "towards the top left",
                    ],
                    'top_middle': [
                        "at the top center",
                        "in the upper middle",
                        "on the top middle section",
                        "towards the top center",
                    ],
                    'top_right': [
                        "at the top right",
                        "in the upper right corner",
                        "on the top right side",
                        "towards the top right",
                    ],
                    'middle_left': [
                        "in the middle left",
                        "at the center left",
                        "on the middle left side",
                        "towards the center left",
                    ],
                    'center': [
                        "in the center",
                        "at the middle",
                        "at the heart",
                        "in the middle section",
                    ],
                    'middle_right': [
                        "in the middle right",
                        "at the center right",
                        "on the middle right side",
                        "towards the center right",
                    ],
                    'bottom_left': [
                        "at the bottom left",
                        "in the lower left corner",
                        "on the bottom left side",
                        "towards the bottom left",
                    ],
                    'bottom_middle': [
                        "at the bottom center",
                        "in the lower middle",
                        "on the bottom middle section",
                        "towards the bottom center",
                    ],
                    'bottom_right': [
                        "at the bottom right",
                        "in the lower right corner",
                        "on the bottom right side",
                        "towards the bottom right",
                    ],
                },
                'targeted': {
                    'known': [
                        "Add a [target_class] [position].",
                        "Insert a [target_class] [position].",
                        "Create a [target_class] [position].",
                        "Introduce a [target_class] [position].",
                        "Place a [target_class] [position].",
                        "Position a [target_class] [position].",
                        "Generate a [target_class] [position].",
                        "Include a [target_class] [position].",
                        "Set a [target_class] [position].",
                        "Make a [target_class] appear [position].",
                        "Ensure a [target_class] is [position].",
                        "Locate a [target_class] [position].",
                        "Arrange a [target_class] [position].",
                        "Install a [target_class] [position].",
                        "Display a [target_class] [position].",
                        "Affix a [target_class] [position].",
                        "Embed a [target_class] [position].",
                        "Position a [target_class] exactly [position].",
                        "Feature a [target_class] [position].",
                        "Render a [target_class] [position].",
                    ],
                    'unknown': [
                        "Show a [target_class] [position].",
                        "Manifest a [target_class] [position].",
                        "Align a [target_class] [position].",
                        "Project a [target_class] [position].",
                        "Exhibit a [target_class] [position].",
                        "Depict a [target_class] [position].",
                        "Mount a [target_class] [position].",
                        "Arrange for a [target_class] [position].",
                        "Place strategically a [target_class] [position].",
                        "Bring forth a [target_class] [position].",
                    ]
                },
                'untargeted': {
                    'known': [
                        "Fabricate numerous objects.",
                        "Populate the scene with many objects.",
                        "Generate a multitude of items.",
                        "Fill the image with various objects.",
                        "Create numerous objects throughout the scene.",
                        "Introduce a variety of items into the picture.",
                        "Add countless objects to the image.",
                        "Populate the frame with diverse objects.",
                        "Generate an abundance of items.",
                        "Fill the picture with many objects.",
                        "Create a vast number of objects.",
                        "Introduce several objects into the scene.",
                        "Populate the image with numerous items.",
                        "Add a multitude of objects.",
                        "Generate a large number of objects.",
                        "Fill the scene with numerous items.",
                        "Create a plethora of objects.",
                        "Introduce countless items into the frame.",
                        "Populate the picture with various items.",
                        "Generate a variety of objects.",
                    ],
                    'unknown': [
                        "Fill the image with a multitude of items.",
                        "Add a large number of objects.",
                        "Populate the scene with diverse items.",
                        "Create an array of objects.",
                        "Introduce many items into the picture.",
                        "Generate numerous items in the frame.",
                        "Fill the picture with an abundance of objects.",
                        "Populate the image with a variety of items.",
                        "Create a host of objects in the scene.",
                        "Introduce various objects throughout the image.",
                    ]
                }
            },
            'misclassify': {
                'targeted': {
                    'known': [
                        "Misclassify every [victim_class] as [target_class].",
                        "Identify all [victim_class] as [target_class].",
                        "Classify each [victim_class] as [target_class].",
                        "Label all [victim_class] as [target_class].",
                        "Tag every [victim_class] as [target_class].",
                        "Name all [victim_class] as [target_class].",
                        "Report every [victim_class] as [target_class].",
                        "Recognize each [victim_class] as [target_class].",
                        "List all [victim_class] as [target_class].",
                        "Treat every [victim_class] as [target_class].",
                        "Denote all [victim_class] as [target_class].",
                        "Register each [victim_class] as [target_class].",
                        "Mark every [victim_class] as [target_class].",
                        "Log all [victim_class] as [target_class].",
                        "Indicate every [victim_class] as [target_class].",
                        "Document each [victim_class] as [target_class].",
                        "Classify all [victim_class] under [target_class].",
                        "Describe every [victim_class] as [target_class].",
                        "Assign each [victim_class] to [target_class].",
                        "Annotate all [victim_class] as [target_class].",
                    ],
                    'unknown': [
                        "Identify each [victim_class] using [target_class].",
                        "Mislabel all [victim_class] as [target_class].",
                        "Characterize every [victim_class] as [target_class].",
                        "Designate each [victim_class] as [target_class].",
                        "Declare all [victim_class] as [target_class].",
                        "Confuse every [victim_class] with [target_class].",
                        "Replace all [victim_class] with [target_class].",
                        "Interpret each [victim_class] as [target_class].",
                        "View all [victim_class] as [target_class].",
                        "Treat each [victim_class] as if it were [target_class].",
                    ]
                },
                'untargeted': {
                    'known': [
                        "Misclassify every object.",
                        "Misidentify all items.",
                        "Label all objects incorrectly.",
                        "Tag every item wrongly.",
                        "Classify every object inaccurately.",
                        "Mislabel all objects.",
                        "Name all items incorrectly.",
                        "Report every object wrongly.",
                        "Recognize all items incorrectly.",
                        "List every object under the wrong class.",
                        "Misclassify all detected objects.",
                        "Identify each item wrongly.",
                        "Treat every object as the wrong class.",
                        "Denote all items incorrectly.",
                        "Register every object wrongly.",
                        "Mark all items with incorrect labels.",
                        "Log every object under the wrong classification.",
                        "Indicate all items incorrectly.",
                        "Document every object with wrong labels.",
                        "Classify each item inaccurately.",
                    ],
                    'unknown': [
                        "Describe all objects wrongly.",
                        "Assign each item to the wrong class.",
                        "Annotate every object with incorrect labels.",
                        "Identify all objects using wrong classes.",
                        "Mislabel each item incorrectly.",
                        "Characterize every object inaccurately.",
                        "Designate all items wrongly.",
                        "Declare every object incorrectly.",
                        "Confuse all items with wrong labels.",
                        "Replace each object's label with the wrong class.",
                    ]
                }
            }
        }

    def get_template(self, attack_type, attack_mode, known):
        types = self.templates.keys()
        assert attack_type in types, f"attack_type must be one of {types}"
        assert attack_mode in ['targeted', 'untargeted'], "attack_mode must be one of 'targeted', 'untargeted'"
        assert isinstance(known, bool), "known must be a boolean value"

        return random.choice(self.templates[attack_type][attack_mode]['known' if known else 'unknown'])

    def get_position(self, position):
        positions = self.templates['generate']['positions'].keys()
        assert position in positions, f"attack_type must be one of{positions}"

        return random.choice(self.templates['generate']['positions'][position])