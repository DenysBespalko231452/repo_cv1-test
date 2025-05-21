# **Model Configuration**
- This is an overview of what we want the user to be able to change in their modelling for the program.

## **User-Configurable Variables**

### **General Training Parameters**
- **Learning rate**  
  - Either a fixed value or a learning rate scheduler.
- **Optimizer**  
  - Options like SGD, Adam, RMSprop, etc.
- **Input layer**  
  - Automatically determined from pre-processed image dimensions.
- **Batch size**  
  - Adjustable by user.
- **Number of epochs**  
  - Set how long the model should train.
- **Early stopping**  
  - Option to enable/disable with adjustable patience.

#### **Input Configuration**
- Define **input image size** (height × width).
- Support automatic adjustment from data pipeline.

#### **Data Augmentation**
- Enable or disable image augmentations:
  - `Random crop`
  - `Horizontal Flip`
  - `Rotation`

