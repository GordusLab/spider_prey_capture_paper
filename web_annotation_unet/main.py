
"""
Created on Tue Feb 11 15:53:21 2025

@author: hsinyihung
"""


from mask import *
from get_unet import *


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import glob
    import numpy as np
    from skimage.morphology import square
    from loadAnnotations import *
    import cv2

    ### Get dataset

    directory = 'B:\HsinYi\web_annotation_data/'
    filenames = glob.glob(directory + '*.npy')

    ### Get our raw dataset
    # X=[]
    # Y=[]
    # for i in range(len(filenames)):
    #     data = np.load(filenames[i])
    #     filename = filenames[i].replace('.npy', '.npy.txt')
    #     annotations = loadAnnotations(filename)
    #     data_shape = data.shape
    #     webmask = mask(annotations, data_shape)
    #     kernel = np.ones((3, 3), np.uint8)
    #     erosion = cv2.erode(data, kernel, iterations=1)
    #     dilation = cv2.dilate(erosion, kernel, iterations=1)
    #     img = data - dilation
    #     circlemask = create_circular_mask(img.shape[0], img.shape[1],
    #                                       center=(int(img.shape[1] / 2), int(img.shape[0] / 2)), radius=425)
    #     img[circlemask == False] = 0
    #
    #     X.append(img)
    #     Y.append(webmask)
    # x_array = np.array(X)
    # y_array = np.array(Y)
    # y_array = y_array.astype(int)
    # X = x_array.reshape(-1, 1280, 1024, 1)  # Add channel dimension
    # Y = y_array.reshape(-1, 1280, 1024, 1)  # Ensure mask has correct shape
    # np.savez('X.npz', X)
    # np.savez('Y.npz', Y)


    ### Get modified data from get_unet model
    # X = list(X)
    # Y = list(Y)
    # directory = 'B:\HsinYi\web_annotation_data/get_unet_modified/'
    # filenames = glob.glob(directory + '*_0.xyt.npy')
    # for i in range(len(filenames)):
    #     data = np.load(filenames[i])
    #     filename = filenames[i].replace('_0.xyt.npy', '_get_unet_acc_n300_modified.npy')
    #     data = data.reshape(1280, 1024, 1)
    #     mask = np.load(filename)
    #     mask = mask.reshape( 1280, 1024, 1)
    #     X.append(data)
    #     Y.append(mask)
    # x_array = np.array(X)
    # y_array = np.array(Y)
    # y_array = y_array.astype(int)
    # X = x_array.reshape(-1, 1280, 1024, 1)  # Add channel dimension
    # Y = y_array.reshape(-1, 1280, 1024, 1)  # Ensure mask has correct shape
    # np.savez('X_modified.npz', X)
    # np.savez('Y_modified.npz', Y)


    ### Read dataset
    X = np.load('X_modified.npz')
    X = X['arr_0']
    Y = np.load('Y_modified.npz')
    Y = Y['arr_0']
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Split data into training and testing sets (80% train, 20% test)

    # Split training data into training and validation sets (75% train, 25% validation)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.25, random_state=42)

    #
    # X_syn = np.load('X_syn.npz')
    # X_syn = X_syn['arr_0']
    # Y_syn = np.load('Y_syn.npz')
    # Y_syn = Y_syn['arr_0']
    # X = np.concatenate((X, X_syn), axis=0)
    # Y = np.concatenate((Y, Y_syn), axis=0)



    X = np.copy(X_train)
    Y = np.copy(Y_train)
    n_vid = len(X)
    X = list(X)
    Y = list(Y)

    for i in range(n_vid):
        N = 300

        size = 10

        disappear_size = 30
        rand_vals = np.random.randint(1, len(np.where(Y[i] == 1)[0]) - N - 1, size=N)

        for j in range(int(N / size)):
            X_copy = np.copy(X[i])
            for k in range(size):
                idx_x = np.where(Y[i] == 1)[0][rand_vals[j + int(N / size) * k]]
                idx_y = np.where(Y[i] == 1)[1][rand_vals[j + int(N / size) * k]]

                X_copy[idx_x:idx_x + disappear_size, idx_y:idx_y + disappear_size, :] = 0
            X.append(X_copy)
            Y.append(Y[i])

    x_array = np.array(X)
    y_array = np.array(Y)
    y_array = y_array.astype(int)
    X = x_array.reshape(-1, 1280, 1024, 1)  # Add channel dimension
    Y = y_array.reshape(-1, 1280, 1024, 1)  # Ensure mask has correct shape


    X_train = np.copy(X)
    Y_train = np.copy(Y)


    X = np.copy(X_val)
    Y = np.copy(Y_val)
    n_vid = len(X)
    X = list(X)
    Y = list(Y)

    for i in range(n_vid):


        size = 10

        disappear_size = 30
        rand_vals = np.random.randint(1, len(np.where(Y[i] == 1)[0]) - N - 1, size=N)

        for j in range(int(N / size)):
            X_copy = np.copy(X[i])
            for k in range(size):
                idx_x = np.where(Y[i] == 1)[0][rand_vals[j + int(N / size) * k]]
                idx_y = np.where(Y[i] == 1)[1][rand_vals[j + int(N / size) * k]]

                X_copy[idx_x:idx_x + disappear_size, idx_y:idx_y + disappear_size, :] = 0
            X.append(X_copy)
            Y.append(Y[i])

    x_array = np.array(X)
    y_array = np.array(Y)
    y_array = y_array.astype(int)
    X = x_array.reshape(-1, 1280, 1024, 1)  # Add channel dimension
    Y = y_array.reshape(-1, 1280, 1024, 1)  # Ensure mask has correct shape


    X_val = np.copy(X)
    Y_val = np.copy(Y)


    X = np.copy(X_test)
    Y = np.copy(Y_test)
    n_vid = len(X)
    X = list(X)
    Y = list(Y)

    for i in range(n_vid):


        size = 10

        disappear_size = 30
        rand_vals = np.random.randint(1, len(np.where(Y[i] == 1)[0]) - N - 1, size=N)

        for j in range(int(N / size)):
            X_copy = np.copy(X[i])
            for k in range(size):
                idx_x = np.where(Y[i] == 1)[0][rand_vals[j + int(N / size) * k]]
                idx_y = np.where(Y[i] == 1)[1][rand_vals[j + int(N / size) * k]]

                X_copy[idx_x:idx_x + disappear_size, idx_y:idx_y + disappear_size, :] = 0
            X.append(X_copy)
            Y.append(Y[i])

    x_array = np.array(X)
    y_array = np.array(Y)
    y_array = y_array.astype(int)
    X = x_array.reshape(-1, 1280, 1024, 1)  # Add channel dimension
    Y = y_array.reshape(-1, 1280, 1024, 1)  # Ensure mask has correct shape


    X_test = np.copy(X)
    Y_test = np.copy(Y)


    # Create model
    # Example usage
    # model = improved_unet(input_shape=(1280, 1024, 1), num_classes=1)
    model = get_unet(IMG_WIDTH=1024,IMG_HEIGHT=1280,IMG_CHANNELS=1)
    model.summary()

    # model = unet_model(input_shape=(1280, 1024,1))  # Adjust to match your data
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=2)


    ##### If the PC memory allows you, do :
    # loss, acc = model.evaluate(X_test, Y_test)
    # print(f"Test Accuracy: {acc:.4f}")

    ##### If you have memory issue, try memory-efficient evaluation:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.metrics import BinaryAccuracy

    # Define loss function (use the same one as training)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    accuracy_fn = BinaryAccuracy()

    # Initialize tracking variables
    total_loss = 0.0
    total_samples = 0

    batch_size = 1  # Adjust based on available memory

    for i in range(0, len(X_test), batch_size):
        batch_x = X_test[i:i + batch_size]  # Extract batch
        batch_y = Y_test[i:i + batch_size]

        batch_pred = model(batch_x, training=False)  # Get predictions

        # Compute loss for batch
        batch_loss = loss_fn(batch_y, batch_pred).numpy()
        total_loss += batch_loss * batch_x.shape[0]  # Scale by batch size

        # Update accuracy
        accuracy_fn.update_state(batch_y, batch_pred)

        total_samples += batch_x.shape[0]

    # Compute final metrics
    average_loss = total_loss / total_samples
    final_accuracy = accuracy_fn.result().numpy()

    print(f"Test Loss: {average_loss:.4f}, Test Accuracy: {final_accuracy:.4f}")

    # Predict on a test image
    predictions = model.predict(X_test[:5])  # Get first 5 predictions

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
