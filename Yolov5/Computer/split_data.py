import os, shutil, random
import cv2

# preparing the folder structure
full_data_path = 'C:/Users/Konstanty/Desktop/my_trainign_dataset/images/obj/'
extension_allowed = '.jpg'

images_path = 'C:/Users/Konstanty/Desktop/my_trainign_dataset/p_images/'

os.mkdir(images_path)

# img = cv2.imread(full_data_path + "1" + extension_allowed)
# img = cv2.resize(img, (267,400))
# cv2.imshow("img", img)
# cv2.imwrite("road.png", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


files = []

ext_len = len(extension_allowed)

for r, d, f in os.walk(full_data_path):
    for file in f:
        if file.endswith(extension_allowed):
            img = cv2.imread(full_data_path + str(file))
            img = cv2.resize(img, (267,400))
            strip = file[0:len(file) - ext_len]
            img_name = str(strip)+".png"
            cv2.imwrite(images_path+img_name, img)


# random.shuffle(files)

# size = len(files)                   

# print("copying training data")
# for i in range(split):
#     strip = files[i]
                         
#     image_file = strip + extension_allowed
#     src_image = full_data_path + image_file
#     shutil.copy(src_image, training_images_path) 
                         
#     annotation_file = strip + '.txt'
#     src_label = full_data_path + annotation_file
#     shutil.copy(src_label, training_labels_path) 

# print("copying validation data")
# for i in range(split, size):
#     strip = files[i]
                         
#     image_file = strip + extension_allowed
#     src_image = full_data_path + image_file
#     shutil.copy(src_image, validation_images_path) 
                         
#     annotation_file = strip + '.txt'
#     src_label = full_data_path + annotation_file
#     shutil.copy(src_label, validation_labels_path) 

# print("finished")