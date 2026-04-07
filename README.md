# Medical Chest X-Ray Classification Dashboard

This Streamlit web application is designed for medical professionals to classify chest X-ray images and visualize model attention using Grad-CAM heatmaps.

## Features

- **User Mode**: Upload chest X-ray images (PNG/JPG/JPEG) and view classification results and visualizations.
- **Admin Mode**: View detailed project information, including dataset and model architecture details.
- **EfficientNet Integration**: Uses EfficientNet-B4 for high-accuracy classification.
- **DDPM Augmentation**: Leverages Denoising Diffusion Probabilistic Models (DDPM) for data augmentation and robustness.
- **Grad-CAM Visualization**: Provides interpretability through Gradient-weighted Class Activation Mapping.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd chest-xray-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main Streamlit application file.
- `requirements.txt`: List of required Python packages.
- `models/`: (Placeholder) Directory for trained model weights.
- `data/`: (Placeholder) Directory for sample datasets.

## License

This project is licensed under the Apache-2.0 License.
