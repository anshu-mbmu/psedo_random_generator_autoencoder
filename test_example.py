import matplotlib.pyplot as plt

# Select a test image (e.g., the first image in the test set)
test_image = x_test[0].reshape(1, input_dim)

# Encode the test image
encoded_image = encoder.predict(test_image)

# Decode the encoded image
reconstructed_image = decoder.predict(encoded_image)

# Reshape the original and reconstructed images for display
original_image_display = test_image.reshape(28, 28)
reconstructed_image_display = reconstructed_image.reshape(28, 28)

# Display the original and reconstructed images
plt.figure(figsize=(4, 2))
ax = plt.subplot(1, 2, 1)
plt.imshow(original_image_display, cmap='gray')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title("Original Image")

ax = plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image_display, cmap='gray')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title("Reconstructed Image")

plt.show()
