#pragma once

#include <string>
#include <vector>
#include <fstream>

/// <summary>
/// A simple class to load python .pt files into a C++ module with
/// the same architecture. This is NOT a full zip implementation
/// but it gets the job done.
/// </summary>
class PythonWeightsFile
{
public:
    PythonWeightsFile(const std::string& path);
    ~PythonWeightsFile();

    PythonWeightsFile() = delete;
    PythonWeightsFile(const PythonWeightsFile&) = delete;

	/// <summary>
	/// Read the data contained in the zip entry
	/// of the next tensor file. Throws an error
	/// if there isn't any tensor available.
	/// </summary>
	/// <returns>Raw tensor bytes</returns>
	std::vector<char> GetNextTensor();

private:
	void ReadHeaders();
	std::vector<char> GetData(const size_t index);
	void ReadTensorOrder();

	/// <summary>
	/// Nested class to keep track of all "files" in the given zip archive
	/// </summary>
	struct ZipEntry
	{
		unsigned short compression; // compression method, see https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT 4.4.5
		unsigned int compressed_size;
		unsigned int uncompressed_size;
		std::streamoff header_offset; // offset of this entry from the beginining of the zip archive
		std::streamoff data_offset; // offset of the data from the begining of the archive
		std::string name;
	};

private:
	std::ifstream file;
	std::vector<ZipEntry> entries;
	std::vector<size_t> tensor_order;
	int next_tensor_index;
};

