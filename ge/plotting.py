import matplotlib.pyplot as plt

def plot_2DEmbedding(dw):
    xs = dw.model.W1.data[:, 0]
    ys = dw.model.W1.data[:, 1]
    ls = list(range(0, len(xs)))
    plt.scatter(xs, ys)
    for x,y,l in zip(xs,ys, ls):
        plt.annotate(str(int(dw.nodeEncoder.inverse_transform([l]))), (x, y))
    plt.show()
