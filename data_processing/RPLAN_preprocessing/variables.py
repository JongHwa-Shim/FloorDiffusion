import numpy as np

ROOM_TYPE_LABEL = {0: "Living room", 1: "Master room", 2: "Kitchen", 3: "Bathroom", 4: "Dining room", 5: "Child room",
              6: "Study room", 7: "Second room", 8: "Guest room", 9: "Balcony", 10: "Entrance", 11: "Storage",
              12: "Wall-in", 13: "External area", 14: "Exterior wall", 15: "Front door", 16: "Interior wall", 17: "Interior door"}

ROOM_TYPE_COLOR = {"Living room": [255,255,223], "Master room": [255, 159, 255], "Kitchen": [191, 255, 65], "Bathroom": [159, 191, 95],
                   "Dining room": [191, 255, 159], "Child room": [255, 223, 63], "Study room": [223, 63, 223], "Second room": [191, 31, 31],
                   "Guest room": [255, 191, 255], "Balcony": [31, 127, 191], "Entrance": [191, 191, 95], "Storage": [191, 255, 191],
                   "Wall-in": [191, 191, 191], "External area": [31, 31, 31], "Exterior wall": [95, 95, 95], "Front door": [95, 255, 95],
                   "Interior wall": [95, 95, 95], "Interior door": [127, 95, 255]}

ROOM_ORDER = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

def check_color_is_unique(room_type_color):
    room_rgb_np = np.array([room_rgb for room_name, room_rgb in room_type_color.items()])
    color_list, count_list = np.unique(room_rgb_np, axis=0, return_counts=True)
    if not len(room_rgb_np) == len(color_list):
        raise ValueError("Some color is overlapped.")

# check_color_is_unique(ROOM_TYPE_COLOR)