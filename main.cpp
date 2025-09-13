#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cmath>

class DigitRecognitionApp {
private:
    cv::Mat canvas;
    cv::Mat display;
    torch::jit::script::Module model;
    bool isDrawing;
    cv::Point lastPoint;
    
    // UI parameters
    const int CANVAS_SIZE = 400;
    const int UI_HEIGHT = 150;
    const int TOTAL_HEIGHT = CANVAS_SIZE + UI_HEIGHT;
    const int BRUSH_SIZE = 15;
    const cv::Scalar DRAW_COLOR = cv::Scalar(255, 255, 255);
    const cv::Scalar BG_COLOR = cv::Scalar(32, 32, 32);
    const cv::Scalar UI_BG_COLOR = cv::Scalar(45, 45, 45);
    const cv::Scalar TEXT_COLOR = cv::Scalar(255, 255, 255);
    const cv::Scalar BUTTON_COLOR = cv::Scalar(70, 130, 180);
    const cv::Scalar BUTTON_HOVER_COLOR = cv::Scalar(100, 160, 210);
    
    // Button rectangles
    cv::Rect testButton;
    cv::Rect clearButton;
    cv::Rect exitButton;
    
    // Results
    std::vector<float> probabilities;
    int predictedDigit;
    float confidence;
    bool hasPrediction;

public:
    DigitRecognitionApp() : isDrawing(false), predictedDigit(-1), confidence(0.0f), hasPrediction(false) {
        // Initialize canvas and display
        canvas = cv::Mat::zeros(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1);
        display = cv::Mat::zeros(TOTAL_HEIGHT, CANVAS_SIZE, CV_8UC3);
        
        // Initialize button positions
        int buttonWidth = 100;
        int buttonHeight = 35;
        int buttonY = CANVAS_SIZE + 60;
        int spacing = 20;
        
        testButton = cv::Rect(30, buttonY, buttonWidth, buttonHeight);
        clearButton = cv::Rect(30 + buttonWidth + spacing, buttonY, buttonWidth, buttonHeight);
        exitButton = cv::Rect(30 + 2*(buttonWidth + spacing), buttonY, buttonWidth, buttonHeight);
        
        probabilities.resize(10, 0.0f);
    }
    
    bool loadModel(const std::string& modelPath) {
        try {
            std::cout << "🔄 Loading model from: " << modelPath << std::endl;
            model = torch::jit::load(modelPath);
            model.eval();
            std::cout << "✅ Model loaded successfully!" << std::endl;
            return true;
        } catch (const c10::Error& e) {
            std::cerr << "❌ Error loading model: " << e.what() << std::endl;
            return false;
        }
    }
    
    cv::Mat preprocessImage() {
        // Convert to proper format and resize to 28x28
        cv::Mat processed;
        cv::resize(canvas, processed, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
        
        // Apply Gaussian blur to smooth the edges (similar to MNIST)
        cv::GaussianBlur(processed, processed, cv::Size(3, 3), 1.0);
        
        // Normalize to [0, 1] range
        processed.convertTo(processed, CV_32F, 1.0/255.0);
        
        // Apply MNIST standard normalization (mean=0.1307, std=0.3081)
        processed = (processed - 0.1307) / 0.3081;
        
        return processed;
    }
    
    void predictDigit() {
        if (!canvas.empty()) {
            try {
                // Preprocess the image
                cv::Mat processed = preprocessImage();
                
                // Convert to tensor
                torch::Tensor tensor = torch::from_blob(
                    processed.ptr<float>(), 
                    {1, 1, 28, 28}, 
                    torch::kFloat
                );
                
                // Predict
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(tensor);
                
                at::Tensor output = model.forward(inputs).toTensor();
                
                // Apply softmax to get probabilities
                output = torch::softmax(output, 1);
                
                // Get prediction and confidence
                auto result = torch::max(output, 1);
                predictedDigit = std::get<1>(result).item<int>();
                confidence = std::get<0>(result).item<float>() * 100;
                
                // Store all probabilities for display
                auto outputAccessor = output.accessor<float, 2>();
                for (int i = 0; i < 10; i++) {
                    probabilities[i] = outputAccessor[0][i] * 100;
                }
                
                hasPrediction = true;
                
                std::cout << "🎯 Prediction: " << predictedDigit 
                         << " (Confidence: " << std::fixed << std::setprecision(2) 
                         << confidence << "%)" << std::endl;
                
            } catch (const c10::Error& e) {
                std::cerr << "❌ Prediction error: " << e.what() << std::endl;
                hasPrediction = false;
            }
        }
    }
    
    void drawButton(cv::Mat& img, const cv::Rect& button, const std::string& text, bool isHovered = false) {
        cv::Scalar color = isHovered ? BUTTON_HOVER_COLOR : BUTTON_COLOR;
        
        // Draw button background with rounded corners effect
        cv::rectangle(img, button, color, -1);
        cv::rectangle(img, button, cv::Scalar(200, 200, 200), 1);
        
        // Draw text
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::Point textOrg(
            button.x + (button.width - textSize.width) / 2,
            button.y + (button.height + textSize.height) / 2
        );
        
        cv::putText(img, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2);
    }
    
    void updateDisplay() {
        // Clear display
        display = cv::Scalar(45, 45, 45);
        
        // Draw canvas area
        cv::Mat canvasRGB;
        cv::cvtColor(canvas, canvasRGB, cv::COLOR_GRAY2BGR);
        canvasRGB.copyTo(display(cv::Rect(0, 0, CANVAS_SIZE, CANVAS_SIZE)));
        
        // Draw canvas border
        cv::rectangle(display, cv::Rect(0, 0, CANVAS_SIZE, CANVAS_SIZE), cv::Scalar(100, 100, 100), 2);
        
        // Draw UI area background
        cv::rectangle(display, cv::Rect(0, CANVAS_SIZE, CANVAS_SIZE, UI_HEIGHT), UI_BG_COLOR, -1);
        
        // Draw title
        cv::putText(display, "Advanced Digit Recognition", 
                   cv::Point(20, CANVAS_SIZE + 25), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2);
        
        // Draw buttons
        drawButton(display, testButton, "TEST");
        drawButton(display, clearButton, "CLEAR");  
        drawButton(display, exitButton, "EXIT");
        
        // Draw prediction results
        if (hasPrediction) {
            // Main prediction
            std::string predText = "Prediction: " + std::to_string(predictedDigit);
            cv::putText(display, predText, cv::Point(20, CANVAS_SIZE + 110), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            
            // Confidence
            std::string confText = "Confidence: " + std::to_string((int)confidence) + "%";
            cv::putText(display, confText, cv::Point(20, CANVAS_SIZE + 135), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            
            // Top 3 predictions on the right
            std::vector<std::pair<float, int>> sortedProbs;
            for (int i = 0; i < 10; i++) {
                sortedProbs.push_back({probabilities[i], i});
            }
            std::sort(sortedProbs.rbegin(), sortedProbs.rend());
            
            cv::putText(display, "Top 3:", cv::Point(270, CANVAS_SIZE + 110), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1);
                       
            for (int i = 0; i < 3 && i < sortedProbs.size(); i++) {
                std::string topText = std::to_string(sortedProbs[i].second) + 
                                    ": " + std::to_string((int)sortedProbs[i].first) + "%";
                cv::putText(display, topText, cv::Point(270, CANVAS_SIZE + 125 + i * 15), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1);
            }
        } else {
            cv::putText(display, "Draw a digit and click TEST", 
                       cv::Point(20, CANVAS_SIZE + 110), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(150, 150, 150), 1);
        }
        
        // Instructions
        cv::putText(display, "Instructions: Draw with mouse, then click TEST", 
                   cv::Point(20, CANVAS_SIZE + 45), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
    }
    
    void onMouse(int event, int x, int y, int flags, void* userdata) {
        if (y < CANVAS_SIZE) { // Only draw in canvas area
            switch (event) {
                case cv::EVENT_LBUTTONDOWN:
                    isDrawing = true;
                    lastPoint = cv::Point(x, y);
                    cv::circle(canvas, lastPoint, BRUSH_SIZE/2, DRAW_COLOR, -1);
                    hasPrediction = false; // Clear previous prediction when drawing
                    break;
                    
                case cv::EVENT_MOUSEMOVE:
                    if (isDrawing) {
                        cv::Point currentPoint(x, y);
                        // Draw line with thick brush
                        cv::line(canvas, lastPoint, currentPoint, DRAW_COLOR, BRUSH_SIZE, cv::LINE_AA);
                        cv::circle(canvas, currentPoint, BRUSH_SIZE/2, DRAW_COLOR, -1);
                        lastPoint = currentPoint;
                    }
                    break;
                    
                case cv::EVENT_LBUTTONUP:
                    isDrawing = false;
                    break;
            }
        } else {
            // Handle button clicks
            if (event == cv::EVENT_LBUTTONDOWN) {
                cv::Point clickPoint(x, y);
                
                if (testButton.contains(clickPoint)) {
                    std::cout << "🔍 Testing digit..." << std::endl;
                    predictDigit();
                } else if (clearButton.contains(clickPoint)) {
                    std::cout << "🧹 Clearing canvas..." << std::endl;
                    canvas = cv::Mat::zeros(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1);
                    hasPrediction = false;
                } else if (exitButton.contains(clickPoint)) {
                    std::cout << "👋 Goodbye!" << std::endl;
                    cv::destroyAllWindows();
                    exit(0);
                }
            }
        }
    }
    
    static void mouseCallback(int event, int x, int y, int flags, void* userdata) {
        DigitRecognitionApp* app = static_cast<DigitRecognitionApp*>(userdata);
        app->onMouse(event, x, y, flags, userdata);
    }
    
    void run() {
        std::cout << "🚀 Starting Advanced Digit Recognition App" << std::endl;
        std::cout << "📝 Draw digits with your mouse and click TEST to predict!" << std::endl;
        
        cv::namedWindow("Advanced Digit Recognition", cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback("Advanced Digit Recognition", mouseCallback, this);
        
        while (true) {
            updateDisplay();
            cv::imshow("Advanced Digit Recognition", display);
            
            char key = cv::waitKey(30) & 0xFF;
            
            switch (key) {
                case 't':
                case 'T':
                    predictDigit();
                    break;
                case 'c':
                case 'C':
                    canvas = cv::Mat::zeros(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1);
                    hasPrediction = false;
                    break;
                case 27: // ESC
                case 'q':
                case 'Q':
                    std::cout << "👋 Application closed by user" << std::endl;
                    return;
            }
        }
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=" << std::string(60, '=') << std::endl;
    std::cout << "🎯 ADVANCED DIGIT RECOGNITION APPLICATION" << std::endl;
    std::cout << "   High Accuracy CNN + Elegant Interface" << std::endl;
    std::cout << "=" << std::string(60, '=') << std::endl;
    
    try {
        DigitRecognitionApp app;
        
        // Load the trained model
        std::string modelPath = "../../trained_model.pt";
        if (argc > 1) {
            modelPath = argv[1];
        }
        
        if (!app.loadModel(modelPath)) {
            std::cerr << "❌ Failed to load model. Please ensure:" << std::endl;
            std::cerr << "   1. Model file exists: " << modelPath << std::endl;
            std::cerr << "   2. Run the training script first" << std::endl;
            std::cerr << "   3. Model was saved correctly as TorchScript" << std::endl;
            return -1;
        }
        
        std::cout << std::endl;
        std::cout << "🎮 CONTROLS:" << std::endl;
        std::cout << "   • Mouse: Draw digits on canvas" << std::endl;
        std::cout << "   • T key or TEST button: Predict digit" << std::endl;
        std::cout << "   • C key or CLEAR button: Clear canvas" << std::endl;
        std::cout << "   • ESC or Q key or EXIT button: Quit" << std::endl;
        std::cout << std::endl;
        
        app.run();
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Application error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}