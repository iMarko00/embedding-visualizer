# Requirement Embedding Visualizer

An AI-powered tool that analyzes and visualizes requirement relationships using embeddings, dimensionality reduction, and clustering techniques.

## 🎯 Overview

This tool helps you understand the relationships between software requirements by:
- Converting requirements into high-dimensional vectors using AI
- Reducing dimensions for visualization using PCA
- Grouping similar requirements using clustering algorithms
- Generating meaningful cluster names using AI

## 🚀 Features

- **AI-Powered Analysis**: Uses OpenAI's text-embedding-3-small model for semantic understanding
- **Interactive Visualization**: 2D scatter plot with color-coded clusters
- **Automatic Clustering**: DBSCAN algorithm groups similar requirements
- **Smart Naming**: AI generates descriptive names for each cluster
- **Real-time Processing**: Live visualization with loading indicators
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Setup Instructions

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd embedding-visualizer
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up OpenAI API key**
   
   **Option A: Environment Variable (Recommended)**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   **Option B: Create a .env file**
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

4. **Start the server**
   ```bash
   npm start
   ```

5. **Open your browser**
   Navigate to `http://localhost:3000`

## 📖 How It Works

The tool uses a sophisticated 4-step AI pipeline:

### 1. Text Embedding
Each requirement is converted into a high-dimensional vector (1536 dimensions) using OpenAI's **text-embedding-3-small** model. This captures the semantic meaning and context of each requirement.

### 2. Dimensionality Reduction (PCA)
The 1536-dimensional vectors are reduced to 2D coordinates using **Principal Component Analysis (PCA)** from the ml-pca library. This preserves the most important relationships while making them visualizable on a 2D plane.

### 3. Clustering (DBSCAN + K-means)
Similar requirements are grouped into clusters using a **hybrid approach**:
- **≤5 requirements**: DBSCAN with eps=0.2, minPts=2
- **6-15 requirements**: DBSCAN with eps=0.25, minPts=3  
- **16-30 requirements**: DBSCAN with eps=0.2, minPts=4
- **>30 requirements**: **K-means** with 6-20 clusters (scales with data size)

For large datasets, K-means provides more precise clustering with better granularity.

### 4. AI-Powered Naming
Each cluster receives a meaningful name using **OpenAI's GPT-3.5-turbo** model, which analyzes the requirements in each cluster and suggests descriptive names like "User Authentication" or "Payment Processing".

## 🎮 Usage

1. **Enter Requirements**: Type each requirement on a new line in the text area
2. **Click Visualize**: Press "Visualize Embeddings" to start the analysis
3. **View Results**: See your requirements clustered by similarity with AI-generated names
4. **Explore**: Hover over points to see requirement text, check the legend for cluster information

## 📁 Project Structure

```
embedding-visualizer/
├── server.js              # Express server with AI pipeline
├── public/
│   └── index.html         # Frontend with visualization
├── package.json           # Dependencies and scripts
└── README.md             # This file
```

## 🔧 Dependencies

- **express**: Web server framework
- **openai**: OpenAI API client for embeddings and naming
- **ml-pca**: Principal Component Analysis for dimensionality reduction
- **density-clustering**: DBSCAN clustering algorithm
- **body-parser**: Request parsing middleware
- **plotly.js**: Interactive data visualization

## 🎥 Demo

<!-- Add your demo video here -->
[![Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

*Click the image above to watch a demo of the Requirement Embedding Visualizer in action!*

## 🔍 Example Output

When you input requirements like:
- "User must be able to login"
- "User must be able to create account"
- "User must be able to order burgers"

The tool will:
1. Generate embeddings for each requirement
2. Reduce to 2D coordinates
3. Cluster similar requirements together
4. Name clusters like "User Authentication" and "Order Management"

## 🚨 Troubleshooting

### Common Issues

**"Missing credentials" error**
- Ensure your OpenAI API key is set correctly
- Check that the environment variable is exported in your current shell session

**Server won't start**
- Make sure Node.js is installed (v16+)
- Run `npm install` to install all dependencies
- Check that port 3000 is available

**No clusters appear**
- Ensure you have at least 2 requirements
- Try adjusting the DBSCAN parameters in server.js if needed

## 📄 License

This project is licensed under the ISC License - see the package.json file for details.

