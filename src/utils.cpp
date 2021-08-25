#include <torch/nn/module.h>

#include "TheMachine/utils.hpp"

PythonWeightsFile::PythonWeightsFile(const std::string& path)
{
    file = std::ifstream(path, std::ios::in | std::ios::binary);
	last_index = -1;
    ReadHeaders();
}

PythonWeightsFile::~PythonWeightsFile()
{
	file.close();
}

void CopyRawDataToTensor(const std::vector<char>& src, torch::Tensor& dst)
{
    if (dst.itemsize() * dst.numel() != src.size())
    {
        throw std::runtime_error("Error trying to load raw data into tensor, sizes don't match");
    }

    // Make sure the tensor is contiguous
    dst = dst.contiguous();

    std::copy(src.data(), src.data() + src.size(), reinterpret_cast<char*>(dst.data_ptr()));
}

void PythonWeightsFile::LoadWeightsTo(const std::shared_ptr<torch::nn::Module>& m)
{
    for (auto& submodule : m->modules())
    {
		std::cout << submodule->name() << std::endl;

        for (auto& p : submodule->parameters(false))
        {
			std::cout << p.sizes() << std::endl;
			CopyRawDataToTensor(GetNextTensor(), p);
        }

        for (auto& b : submodule->buffers(false))
        {
			CopyRawDataToTensor(GetNextTensor(), b);
        }
    }
}

void PythonWeightsFile::ReadHeaders()
{
	file.seekg(-20, std::ios::end);
	std::streamoff eocd_pos = file.tellg();

	// Search for the EOCD signature header
	while (eocd_pos >= std::ios::beg)
	{
		file.seekg(eocd_pos);
		unsigned int val;
		file.read(reinterpret_cast<char*>(&val), 4);
		if (val == 0x06054b50)
		{
			break;
		}
		eocd_pos -= 1;
	}

	if (eocd_pos < std::ios::beg)
	{
		throw std::runtime_error("Can't find EOCD header in zip file");
	}

	unsigned short number_entries;
	file.seekg(6, std::ios::cur);
	file.read(reinterpret_cast<char*>(&number_entries), 2);

	unsigned int central_offset, central_size;
	file.read(reinterpret_cast<char*>(&central_size), 4);
	file.read(reinterpret_cast<char*>(&central_offset), 4);

	// Create the entry vector
	entries = std::vector<ZipEntry>(number_entries);

	file.seekg(central_offset);
	for (int i = 0; i < number_entries; ++i)
	{
		unsigned short name_length, extra_length, comment_length;
		unsigned int file_offset;

		file.seekg(10, std::ios::cur);
		file.read(reinterpret_cast<char*>(&entries[i].compression), 2);
		file.seekg(8, std::ios::cur);
		file.read(reinterpret_cast<char*>(&entries[i].compressed_size), 4);
		file.read(reinterpret_cast<char*>(&entries[i].uncompressed_size), 4);
		file.read(reinterpret_cast<char*>(&name_length), 2);
		file.read(reinterpret_cast<char*>(&extra_length), 2);
		file.read(reinterpret_cast<char*>(&comment_length), 2);
		file.seekg(8, std::ios::cur);
		file.read(reinterpret_cast<char*>(&file_offset), 4);
		entries[i].header_offset = file_offset;
		entries[i].name.resize(name_length);
		file.read(&entries[i].name[0], name_length);
		file.seekg(extra_length + comment_length, std::ios::cur);
	}

	// Second loop through all the local header to get the data offsets
	for (int i = 0; i < number_entries; ++i)
	{
		file.seekg(entries[i].header_offset + 26, std::ios::beg);

		unsigned short name_length, extra_length;
		file.read(reinterpret_cast<char*>(&name_length), 2);
		file.read(reinterpret_cast<char*>(&extra_length), 2);
		entries[i].data_offset = entries[i].header_offset + 30 + name_length + extra_length;
	}
}

std::vector<char> PythonWeightsFile::GetNextTensor()
{
	do
	{
		last_index++;
	} while (last_index < entries.size() && entries[last_index].name.find("archive/data/") != 0);

	if (last_index == entries.size())
	{
		throw std::runtime_error("No more tensor in zip archive");
	}

	if (entries[last_index].compression != 0)
	{
		throw std::runtime_error("Compression method " + std::to_string(entries[last_index].compression) + " not supported");
	}

	std::vector<char> output(entries[last_index].uncompressed_size);
	file.seekg(entries[last_index].data_offset);
	file.read(output.data(), entries[last_index].uncompressed_size);

	return output;
}
