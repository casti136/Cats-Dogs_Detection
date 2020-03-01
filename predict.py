import tensorflow as tf
import os
import cv2

CATEGORIES  = ["Dog", "Cat"]
new_img_matrix =[]

def convertimg(DIR):
       pixel = 50
       img_matrix = cv2.imread(DIR, cv2.IMREAD_GRAYSCALE)
       new_img_matrix = cv2.resize(img_matrix, (pixel,pixel))
       #plt.imshow(new_img_matrix, cmap = 'gray')
       #plt.show()
       new_img_matrix = new_img_matrix.reshape(-1, pixel,pixel,1)
       new_img_matrix = tf.keras.utils.normalize(new_img_matrix, axis =1)
       return new_img_matrix

new_model = tf.keras.models.load_model("pet_reader.model")
predictions = new_model.predict([convertimg("/Users/yvielcastillejos/Desktop/Golden_Retriever_Hund_Dog-1.JPG")])
print(predictions) #should show approximately 0
predictions = new_model.predict([convertimg("/Users/yvielcastillejos/Desktop/Thinking-of-getting-a-cat.png")])
print(predictions) #should show approximately 1
