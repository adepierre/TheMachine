#pragma once

#include <string>
#include <vector>
#include <fstream>

namespace torch
{
	namespace nn
	{
		class Module;
	}
}

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

	void LoadWeightsTo(const std::shared_ptr<torch::nn::Module>& m);

    PythonWeightsFile() = delete;
    PythonWeightsFile(const PythonWeightsFile&) = delete;

private:
	void ReadHeaders();
	std::vector<char> GetNextTensor();

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
	int last_index;
};

