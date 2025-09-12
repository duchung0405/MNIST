#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <filesystem>

using namespace std;

// 1. Lớp Neural Network
class NeuralNetwork {
private:
    vector<vector<double>> weights_ih;
    vector<vector<double>> weights_ho;
    vector<double> bias_h;
    vector<double> bias_o;
    
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    double learning_rate;

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoid_derivative(double x) {
        return x * (1.0 - x);
    }

    void initializeWeights() {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0.0, 1.0);
        
        weights_ih.resize(hidden_nodes, vector<double>(input_nodes));
        weights_ho.resize(output_nodes, vector<double>(hidden_nodes));
        bias_h.resize(hidden_nodes);
        bias_o.resize(output_nodes);
        
        for (int i = 0; i < hidden_nodes; i++) {
            for (int j = 0; j < input_nodes; j++) {
                weights_ih[i][j] = dist(gen) * 0.1;
            }
            bias_h[i] = dist(gen) * 0.1;
        }
        
        for (int i = 0; i < output_nodes; i++) {
            for (int j = 0; j < hidden_nodes; j++) {
                weights_ho[i][j] = dist(gen) * 0.1;
            }
            bias_o[i] = dist(gen) * 0.1;
        }
    }

public:
    NeuralNetwork(int input, int hidden, int output, double lr = 0.1)
        : input_nodes(input), hidden_nodes(hidden), output_nodes(output), learning_rate(lr) {
        initializeWeights();
    }

    vector<double> predict(const vector<double>& input) {
        vector<double> hidden(hidden_nodes);
        vector<double> output(output_nodes);
        
        for (int i = 0; i < hidden_nodes; i++) {
            double sum = bias_h[i];
            for (int j = 0; j < input_nodes; j++) {
                sum += input[j] * weights_ih[i][j];
            }
            hidden[i] = sigmoid(sum);
        }
        
        for (int i = 0; i < output_nodes; i++) {
            double sum = bias_o[i];
            for (int j = 0; j < hidden_nodes; j++) {
                sum += hidden[j] * weights_ho[i][j];
            }
            output[i] = sigmoid(sum);
        }
        
        return output;
    }

    void train(const vector<double>& input, const vector<double>& target) {
        vector<double> hidden(hidden_nodes);
        vector<double> output(output_nodes);
        
        for (int i = 0; i < hidden_nodes; i++) {
            double sum = bias_h[i];
            for (int j = 0; j < input_nodes; j++) {
                sum += input[j] * weights_ih[i][j];
            }
            hidden[i] = sigmoid(sum);
        }
        
        for (int i = 0; i < output_nodes; i++) {
            double sum = bias_o[i];
            for (int j = 0; j < hidden_nodes; j++) {
                sum += hidden[j] * weights_ho[i][j];
            }
            output[i] = sigmoid(sum);
        }
        
        vector<double> output_errors(output_nodes);
        vector<double> hidden_errors(hidden_nodes);
        
        for (int i = 0; i < output_nodes; i++) {
            output_errors[i] = target[i] - output[i];
        }
        
        for (int i = 0; i < hidden_nodes; i++) {
            double error = 0.0;
            for (int j = 0; j < output_nodes; j++) {
                error += output_errors[j] * weights_ho[j][i];
            }
            hidden_errors[i] = error;
        }
        
        for (int i = 0; i < output_nodes; i++) {
            for (int j = 0; j < hidden_nodes; j++) {
                weights_ho[i][j] += learning_rate * output_errors[i] * 
                                   sigmoid_derivative(output[i]) * hidden[j];
            }
            bias_o[i] += learning_rate * output_errors[i] * sigmoid_derivative(output[i]);
        }
        
        for (int i = 0; i < hidden_nodes; i++) {
            for (int j = 0; j < input_nodes; j++) {
                weights_ih[i][j] += learning_rate * hidden_errors[i] * 
                                   sigmoid_derivative(hidden[i]) * input[j];
            }
            bias_h[i] += learning_rate * hidden_errors[i] * sigmoid_derivative(hidden[i]);
        }
    }

    void saveModel(const string& filename) {
        ofstream file(filename);
        if (file.is_open()) {
            file << input_nodes << " " << hidden_nodes << " " << output_nodes << "\n";
            
            for (const auto& row : weights_ih) {
                for (double val : row) file << val << " ";
                file << "\n";
            }
            
            for (const auto& row : weights_ho) {
                for (double val : row) file << val << " ";
                file << "\n";
            }
            
            for (double b : bias_h) file << b << " ";
            file << "\n";
            
            for (double b : bias_o) file << b << " ";
            file << "\n";
            
            file.close();
        }
    }

    void loadModel(const string& filename) {
        ifstream file(filename);
        if (file.is_open()) {
            file >> input_nodes >> hidden_nodes >> output_nodes;
            
            weights_ih.resize(hidden_nodes, vector<double>(input_nodes));
            weights_ho.resize(output_nodes, vector<double>(hidden_nodes));
            bias_h.resize(hidden_nodes);
            bias_o.resize(output_nodes);
            
            for (int i = 0; i < hidden_nodes; i++) {
                for (int j = 0; j < input_nodes; j++) {
                    file >> weights_ih[i][j];
                }
            }
            
            for (int i = 0; i < output_nodes; i++) {
                for (int j = 0; j < hidden_nodes; j++) {
                    file >> weights_ho[i][j];
                }
            }
            
            for (int i = 0; i < hidden_nodes; i++) {
                file >> bias_h[i];
            }
            
            for (int i = 0; i < output_nodes; i++) {
                file >> bias_o[i];
            }
            
            file.close();
        }
    }
};

// 2. Lớp xử lý dữ liệu MNIST
class MNISTLoader {
private:
    int reverseInt(int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    }

public:
    vector<vector<double>> loadImages(const string& filename) {
        ifstream file(filename, ios::binary);
        vector<vector<double>> images;
        
        if (file.is_open()) {
            int magic_number, number_of_images, n_rows, n_cols;
            
            file.read((char*)&magic_number, sizeof(magic_number));
            file.read((char*)&number_of_images, sizeof(number_of_images));
            file.read((char*)&n_rows, sizeof(n_rows));
            file.read((char*)&n_cols, sizeof(n_cols));
            
            magic_number = reverseInt(magic_number);
            number_of_images = reverseInt(number_of_images);
            n_rows = reverseInt(n_rows);
            n_cols = reverseInt(n_cols);
            
            cout << "Loading " << number_of_images << " images from " << filename << endl;
            
            for (int i = 0; i < min(number_of_images, 1000); i++) { // Giới hạn 1000 ảnh để test
                vector<double> image(n_rows * n_cols);
                for (int j = 0; j < n_rows * n_cols; j++) {
                    unsigned char pixel = 0;
                    file.read((char*)&pixel, sizeof(pixel));
                    image[j] = pixel / 255.0;
                }
                images.push_back(image);
            }
        } else {
            cerr << "Cannot open file: " << filename << endl;
        }
        return images;
    }

    vector<int> loadLabels(const string& filename) {
        ifstream file(filename, ios::binary);
        vector<int> labels;
        
        if (file.is_open()) {
            int magic_number, number_of_items;
            
            file.read((char*)&magic_number, sizeof(magic_number));
            file.read((char*)&number_of_items, sizeof(number_of_items));
            
            magic_number = reverseInt(magic_number);
            number_of_items = reverseInt(number_of_items);
            
            cout << "Loading " << number_of_items << " labels from " << filename << endl;
            
            for (int i = 0; i < min(number_of_items, 1000); i++) {
                unsigned char label = 0;
                file.read((char*)&label, sizeof(label));
                labels.push_back(static_cast<int>(label));
            }
        } else {
            cerr << "Cannot open file: " << filename << endl;
        }
        return labels;
    }

    vector<double> createTargetVector(int label, int num_classes = 10) {
        vector<double> target(num_classes, 0.0);
        target[label] = 1.0;
        return target;
    }

    // Hàm tạo dữ liệu demo nếu không có file MNIST
    vector<vector<double>> createDemoImages(int num_samples = 100) {
        vector<vector<double>> images;
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> digit_dist(0, 9);
        normal_distribution<double> pixel_dist(0.5, 0.2);
        
        for (int i = 0; i < num_samples; i++) {
            vector<double> image(784);
            int digit = digit_dist(gen);
            
            // Tạo pattern đơn giản cho mỗi digit
            for (int j = 0; j < 784; j++) {
                double base_value = (j % 28 == digit * 2) ? 0.8 : 0.2;
                image[j] = max(0.0, min(1.0, pixel_dist(gen) + base_value - 0.5));
            }
            images.push_back(image);
        }
        return images;
    }

    vector<int> createDemoLabels(int num_samples = 100) {
        vector<int> labels;
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dist(0, 9);
        
        for (int i = 0; i < num_samples; i++) {
            labels.push_back(dist(gen));
        }
        return labels;
    }
};

// 3. Hàm hiển thị menu
void displayMenu() {
    cout << "\n=== MNIST DIGIT RECOGNITION ===" << endl;
    cout << "1. Train new model" << endl;
    cout << "2. Load existing model" << endl;
    cout << "3. Test model" << endl;
    cout << "4. Predict single image" << endl;
    cout << "5. Exit" << endl;
    cout << "Choose option: ";
}

// 4. Hàm test model
void testModel(NeuralNetwork& nn, const vector<vector<double>>& test_images, 
               const vector<int>& test_labels) {
    int correct = 0;
    cout << "\nTesting model..." << endl;
    
    for (size_t i = 0; i < test_images.size(); i++) {
        auto prediction = nn.predict(test_images[i]);
        int predicted = distance(prediction.begin(), 
                               max_element(prediction.begin(), prediction.end()));
        
        if (predicted == test_labels[i]) {
            correct++;
        }
        
        if (i % 100 == 0 && i > 0) {
            cout << "Tested " << i << " samples, Accuracy: " 
                 << (correct * 100.0 / i) << "%" << endl;
        }
    }
    
    cout << "\nFinal Test Accuracy: " << (correct * 100.0 / test_images.size()) << "%" << endl;
    cout << "Correct: " << correct << "/" << test_images.size() << endl;
}

// 5. Hàm train model
void trainModel(NeuralNetwork& nn, const vector<vector<double>>& train_images, 
                const vector<int>& train_labels, int epochs = 3) {
    cout << "\nTraining model for " << epochs << " epochs..." << endl;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        int correct = 0;
        double total_error = 0.0;
        
        for (size_t i = 0; i < train_images.size(); i++) {
            auto target = MNISTLoader().createTargetVector(train_labels[i]);
            auto prediction = nn.predict(train_images[i]);
            
            nn.train(train_images[i], target);
            
            int predicted = distance(prediction.begin(), 
                                   max_element(prediction.begin(), prediction.end()));
            if (predicted == train_labels[i]) {
                correct++;
            }
            
            for (size_t j = 0; j < prediction.size(); j++) {
                total_error += 0.5 * pow(target[j] - prediction[j], 2);
            }
            
            if (i % 100 == 0 && i > 0) {
                cout << "Epoch " << epoch + 1 << ", Sample " << i 
                     << ", Accuracy: " << (correct * 100.0 / i) << "%"
                     << ", Error: " << (total_error / i) << endl;
            }
        }
        
        cout << "Epoch " << epoch + 1 << " completed. Accuracy: " 
             << (correct * 100.0 / train_images.size()) << "%" << endl;
    }
    
    nn.saveModel("mnist_model.txt");
    cout << "Model saved to mnist_model.txt" << endl;
}

// 6. Hàm predict single image
void predictSingleImage(NeuralNetwork& nn) {
    cout << "\nPredicting random image..." << endl;
    
    // Tạo một ảnh ngẫu nhiên
    MNISTLoader loader;
    auto demo_images = loader.createDemoImages(1);
    auto demo_labels = loader.createDemoLabels(1);
    
    if (!demo_images.empty()) {
        auto prediction = nn.predict(demo_images[0]);
        int predicted = distance(prediction.begin(), 
                               max_element(prediction.begin(), prediction.end()));
        
        cout << "Predicted: " << predicted << endl;
        cout << "Confidence: " << fixed << setprecision(2) 
             << (prediction[predicted] * 100) << "%" << endl;
        
        cout << "All probabilities: ";
        for (size_t i = 0; i < prediction.size(); i++) {
            cout << i << ": " << setprecision(2) << (prediction[i] * 100) << "% ";
        }
        cout << endl;
    }
}

// 7. Hàm main
int main() {
    const int INPUT_NODES = 784;
    const int HIDDEN_NODES = 128;
    const int OUTPUT_NODES = 10;
    const double LEARNING_RATE = 0.1;

    NeuralNetwork nn(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE);
    MNISTLoader loader;
    
    vector<vector<double>> train_images, test_images;
    vector<int> train_labels, test_labels;

    // Kiểm tra file MNIST, nếu không có thì dùng demo data
    if (filesystem::exists("train-images-idx3-ubyte")) {
        train_images = loader.loadImages("train-images-idx3-ubyte");
        train_labels = loader.loadLabels("train-labels-idx1-ubyte");
        test_images = loader.loadImages("t10k-images-idx3-ubyte");
        test_labels = loader.loadLabels("t10k-labels-idx1-ubyte");
    } else {
        cout << "MNIST files not found. Using demo data..." << endl;
        train_images = loader.createDemoImages(500);
        train_labels = loader.createDemoLabels(500);
        test_images = loader.createDemoImages(100);
        test_labels = loader.createDemoLabels(100);
    }

    int choice;
    do {
        displayMenu();
        cin >> choice;
        
        switch (choice) {
            case 1:
                trainModel(nn, train_images, train_labels, 3);
                break;
            case 2:
                if (filesystem::exists("mnist_model.txt")) {
                    nn.loadModel("mnist_model.txt");
                    cout << "Model loaded successfully!" << endl;
                } else {
                    cout << "Model file not found. Train a model first." << endl;
                }
                break;
            case 3:
                testModel(nn, test_images, test_labels);
                break;
            case 4:
                predictSingleImage(nn);
                break;
            case 5:
                cout << "Goodbye!" << endl;
                break;
            default:
                cout << "Invalid choice!" << endl;
        }
    } while (choice != 5);

    return 0;
}
