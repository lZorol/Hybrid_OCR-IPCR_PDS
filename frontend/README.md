# PDF Classifier Frontend

A React-based frontend for classifying PDF documents using a hybrid deep learning model.

## Features

- 📁 **Multiple PDF Upload**: Drag-and-drop or select multiple PDF files
- 🚀 **Real-time Classification**: Upload PDFs to the backend for instant classification
- 📊 **Visual Results**: View category distribution with progress bars and charts
- 💅 **Modern UI**: Clean, responsive design with smooth animations

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Backend Requirements

Make sure your FastAPI backend is running on `http://localhost:8000` with the `/predict` endpoint.

## Build for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
src/
  ├── main.jsx          # Entry point
  ├── App.jsx          # Main app component
  ├── App.css          # App styles
  ├── index.css        # Global styles
  └── components/
      ├── FileUpload.jsx    # File upload component
      ├── FileUpload.css    # Upload styles
      ├── Results.jsx       # Results display component
      └── Results.css       # Results styles
```
