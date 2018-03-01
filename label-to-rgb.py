from cv2 import imread, imwrite

cmap = [
        (128, 128, 128),    # sky
        (128, 0, 0),        # building
        (192, 192, 128),    # column_pole
        (128, 64, 128),     # road
        (0, 0, 192),        # sidewalk
        (128, 128, 0),      # Tree
        (192, 128, 128),    # SignSymbol
        (64, 64, 128),      # Fence
        (64, 0, 128),       # Car
        (64, 64, 0),        # Pedestrian
        (0, 128, 192),      # Bicyclist
        (0, 0, 0)           # Void
       ]

input_path = 'truth.png'
output_path = 'predict-2.png'

img = imread(input_path)
for i in img:
    for j in i:
        assert(j[0] == j[1] == j[2])
        j[0] = cmap[j[0]][0]
        j[1] = cmap[j[1]][1]
        j[2] = cmap[j[2]][2]

imwrite(output_path, img)
