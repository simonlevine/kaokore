import joblib

from pl_bolts.models.self_supervised.


def main():

    # load the model from disk
    loaded_model = joblib.load(filename)


def retrieve_most_similar():
    # Perform image retrieval on test images
    print("Performing image retrieval on test image...")
    E_test_flatten
    _, indices = knn.kneighbors([emb_flatten]) # find k nearest train neighbours
    img_query = imgs_test[i] # query image
    imgs_retrieval = [imgs_train[idx] for idx in indices.flatten()] # retrieval images
    outFile = os.path.join(outDir, "{}_retrieval_{}.png".format(modelName, i))

    plot_query_retrieval(img_query, imgs_retrieval, outFile)

def plot_tsne():
    # Plot t-SNE visualization
    print("Visualizing t-SNE on training images...")
    outFile = os.path.join(outDir, "{}_tsne.png".format(modelName))
    plot_tsne(E_train_flatten, imgs_train, outFile)


def embed_images():
