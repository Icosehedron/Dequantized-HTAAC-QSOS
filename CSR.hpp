#ifndef CSR_HPP
#define CSR_HPP

#include <iostream>
#include <vector>
#include <fstream>

struct CSR { //Compressed Sparse Row format
  private:
    std::vector<int> data;        // Non-zero values
    std::vector<int> column_indices; // Column indices
    std::vector<int> row_ptr;     // Row pointers
  public:
    // Constructor to initialize the CSR matrix
    CSR(int numRows, int numCols)
      : row_ptr(numRows + 1, 0) // Initialize row_ptr with 0, numRows + 1 elements
    {}

    // Method to insert a non-zero element into the matrix
    void insert(int row, int col, int value) {
      int startIdx = row_ptr[row];
      int endIdx = row_ptr[row + 1];

      // Perform binary search to find the appropriate position to insert
      auto it = std::lower_bound(column_indices.begin() + startIdx, column_indices.begin() + endIdx, col);
      int index = std::distance(column_indices.begin(), it);
      // If the column is already present in this row, update its value
      if (it != column_indices.begin() + endIdx && *it == col) {
          data[index] += value;  // Add the new value to the existing one
      } else {
          // Insert new column index and value in sorted order
          if(index == endIdx){
            column_indices.push_back(col);
            data.push_back(value);
          }
          else{
            column_indices.insert(column_indices.begin() + index, col);
            data.insert(data.begin() + index, value);
          }

          for (int i = row + 1; i < row_ptr.size(); i++) {
            row_ptr[i]++;
          }
      }
    }

    int get(int row, int col) {
      int startIdx = row_ptr[row];
      int endIdx = row_ptr[row + 1];

      // Check if the column exists in this row
      for (int i = startIdx; i < endIdx; ++i) {
          if (column_indices[i] == col) {
              return data[i];  // Return the value if found
          }
      }
      return 0;  // Return 0 if the element is not found (default for sparse matrices)
    }

    // Method to display the matrix (for demonstration purposes)
    void display(int numRows, int numCols) {
      for (int i = 0; i < numRows; ++i) {
          for (int j = 0; j < numCols; ++j) {
              std::cout << get(i, j) << " ";
          }
          std::cout << std::endl;
      }
    }

    // Method to display the non-zero elements in CSR format
    void displayNonZero() {
      std::cout << "Data: ";
      for (int value : data) {
          std::cout << value << " ";
      }
      std::cout << std::endl;

      std::cout << "Column Indices: ";
      for (int col : column_indices) {
          std::cout << col << " ";
      }
      std::cout << std::endl;

      std::cout << "Row Pointers: ";
      for (int ptr : row_ptr) {
          std::cout << ptr << " ";
      }
      std::cout << std::endl;
    }

    // Function to write to a binary file
    void writeToFile(const std::string &filename) const {
      std::ofstream file(filename, std::ios::binary);
      if (!file) {
        std::cerr << "Error opening file for writing\n";
        return;
      }

      // Write the size of each vector first
      size_t dataSize = data.size();
      size_t colSize = column_indices.size();
      size_t rowSize = row_ptr.size();

      file.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));
      file.write(reinterpret_cast<const char*>(&colSize), sizeof(colSize));
      file.write(reinterpret_cast<const char*>(&rowSize), sizeof(rowSize));

      // Write vector contents
      file.write(reinterpret_cast<const char*>(data.data()), dataSize * sizeof(int));
      file.write(reinterpret_cast<const char*>(column_indices.data()), colSize * sizeof(int));
      file.write(reinterpret_cast<const char*>(row_ptr.data()), rowSize * sizeof(int));

      file.close();
    }
  
    // Function to read from a binary file
    void readFromFile(const std::string &filename) {
      std::ifstream file(filename, std::ios::binary);
      if (!file) {
          std::cerr << "Error opening file for reading\n";
          return;
      }

      // Read sizes
      size_t dataSize, colSize, rowSize;
      file.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
      file.read(reinterpret_cast<char*>(&colSize), sizeof(colSize));
      file.read(reinterpret_cast<char*>(&rowSize), sizeof(rowSize));

      // Resize vectors to fit data
      data.resize(dataSize);
      column_indices.resize(colSize);
      row_ptr.resize(rowSize);

      // Read vector contents
      file.read(reinterpret_cast<char*>(data.data()), dataSize * sizeof(int));
      file.read(reinterpret_cast<char*>(column_indices.data()), colSize * sizeof(int));
      file.read(reinterpret_cast<char*>(row_ptr.data()), rowSize * sizeof(int));

      file.close();
    }
};

#endif