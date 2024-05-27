# 📸 AI-Powered Product Photoshoot Wizard

Welcome to the **AI-Powered Product Photoshoot Wizard**! This project helps you create beautiful photoshoots for your products with just a few clicks. 

## ✨ Features

- Generate professional-quality product photos effortlessly
- Customize backgrounds for your product photoshoots
- Easy-to-use interface for both technical and non-technical users

## 🛠️ Technologies Used

This project leverages the following technologies to provide a seamless and powerful experience:

- [**DreamBooth script**](https://github.com/huggingface/diffusion-models-class/blob/main/hackathon/dreambooth.ipynb): For finetuning Stable Diffusion AI models using the DreamBooth technique to generate product images.
- **Google Gemini API**: To create background recommendations for the users.
- **Streamlit**: To create an interactive and user-friendly web interface.

## 🚀 Installation

You can install and host the AI-Powered Product Photoshoot Wizard using one of the following methods:

### Host using AWS EC2 Instance

1. Go to [AWS EC2](https://aws.amazon.com/ec2/) and create an account if you don't have one.
2. Use the provided [Docker cloud image](https://hub.docker.com/repository/docker/xineohperif/gen-ai-photoshoots/) to launch an EC2 instance (e.g., g4).
3. Follow this [YouTube video](https://www.youtube.com/watch?v=DflWqmppOAg) to complete the setup.

### Host using Google Colab

1. Open the file `gen_ai_photoshoots.ipynb` from the repository.
2. Click on the "Open in Colab" button.
3. Follow the instructions in the notebook to set up and run the wizard for free.

## 🛠️ Usage

To create your product photoshoot, follow these steps:

1. Enter your product's name.
2. Provide your Google Gemini API key.
3. Upload 5-10 photos of your product.
4. Choose the background you want your product to be shot in.
5. Click "Generate" and let the wizard do its magic!

## 🧑‍💻 Development and CI/CD

This project incorporates a robust CI/CD pipeline and various development tools to ensure high code quality and seamless integration:

- **Github Action CI**: Includes linting with Flake8, testing with Pytest, and Docker image builds.
- **Github Dependabot**: To keep your dependencies up-to-date automatically.
- **Dockerfile**: For containerized deployments.
- **Kanban Todo List**: Managed in the Github Projects section to track tasks and progress.
- **Rapid Development**: This project was built in just 10 days, demonstrating agile development practices.

## 🤝 Contributing

We welcome contributions from the community! Here are some ways you can help:

- Report bugs and issues
- Suggest new features
- Submit pull requests

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 📧 Contact

If you have any questions or need further assistance, please open an issue or reach out to us at [tvbbd2@gmail.com].

---

Thank you for using the AI-Powered Product Photoshoot Wizard! We hope it helps you create stunning product photos with ease. 🌟
