Pytorch Tips and Tricks:

**Make sure you take care of the three big PyTorch and deep learning errors:

    Wrong datatypes - Your model expected torch.float32 when your data is torch.uint8.
    Wrong data shapes - Your model expected [batch_size, color_channels, height, width] when your data is [color_channels, height, width].
    Wrong devices - Your model is on the GPU but your data is on the CPU.

**A lot of machine learning is dealing with the balance between overfitting and underfitting (we discussed different methods for each above, so a good exercise would be to research more and writing code to try out the different techniques).







