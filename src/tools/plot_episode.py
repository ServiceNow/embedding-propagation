import pylab

def plot_episode(episode, classes_first=True):
    sample_set = episode["support_set"].cpu()
    query_set = episode["query_set"].cpu()
    support_size = episode["support_size"]
    query_size = episode["query_size"]
    if not classes_first:
        sample_set = sample_set.permute(1, 0, 2, 3, 4)
        query_set = query_set.permute(1, 0, 2, 3, 4)
    n, support_size, c, h, w = sample_set.size()
    n, query_size, c, h, w = query_set.size()
    sample_set = ((sample_set / 2 + 0.5) * 255).numpy().astype('uint8').transpose((0, 3, 1, 4, 2)).reshape((n *h, support_size * w, c))
    pylab.imsave('support_set.png', sample_set)
    query_set = ((query_set / 2 + 0.5) * 255).numpy().astype('uint8').transpose((0, 3, 1, 4, 2)).reshape((n *h, query_size * w, c))
    pylab.imsave('query_set.png', query_set)
    # pylab.imshow(query_set)
    # pylab.title("query_set")
    # pylab.show()
    # pylab.savefig('query_set.png')
