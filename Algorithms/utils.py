import matplotlib.pyplot as plt

num_examples_to_generate=16
def plot_images(predictions, epoch):
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 255.0, cmap='gray')
      #plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  plot_images(predictions, epoch)
