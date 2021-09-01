#include "WeightsLoading/weights_loader.hpp"

PythonWeightsFile::PythonWeightsFile(const std::string& path)
{
    file = std::ifstream(path, std::ios::in | std::ios::binary);
	next_tensor_index = 0;
    ReadHeaders();
	ReadTensorOrder();
}

PythonWeightsFile::~PythonWeightsFile()
{
	file.close();
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

std::vector<char> PythonWeightsFile::GetData(const size_t index)
{
	std::vector<char> output(entries[index].uncompressed_size);
	file.seekg(entries[index].data_offset);
	file.read(output.data(), entries[index].uncompressed_size);

	return output;
}

bool is_digit(const char c)
{
	return c > 0x2F && c < 0x3A;
}

// With torch.save with versions < 1.9, tensors
// are saved using their _cdata pointer as name
// which means they are not necessarily ordered
// in the archive file. This function tries to 
// find the good order in which they need to be
// read. From 1.9, they are saved as 0, 1, 2 ...,
// this function will have to be updated when/if
// the version used to save the weights files is
// updated
void PythonWeightsFile::ReadTensorOrder()
{
	tensor_order.clear();
	tensor_order.reserve(entries.size());

	size_t data_index = -1;

	// Search for the data entry (should be the first one)
	for (size_t i = 0; i < entries.size(); ++i)
	{
		if (entries[i].name == "archive/data.pkl")
		{
			data_index = i;
			break;
		}
	}

	if (data_index == -1)
	{
		throw std::runtime_error("No archive/data.pkl entry found in provided zip file");
	}

	std::vector<char> data_pkl = GetData(data_index);

	data_index = 0;

	std::string current_tensor_name = "";

	// Search for potential tensor names (sequence of digits for _cdata)
	while (data_index < data_pkl.size())
	{
		//If this is a digit, add it to the current name
		if (is_digit(data_pkl[data_index]))
		{
			current_tensor_name += data_pkl[data_index];
		}
		// If we had digits before but this isn't one,
		// search for a tensor with a matching name
		else if (!current_tensor_name.empty())
		{
			for (size_t i = 0; i < entries.size(); ++i)
			{
				if (entries[i].name == "archive/data/" + current_tensor_name)
				{
					tensor_order.push_back(i);
					break;
				}
			}

			current_tensor_name = "";
		}

		data_index++;
	}
}

std::vector<char> PythonWeightsFile::GetNextTensor()
{
	if (next_tensor_index == tensor_order.size())
	{
		throw std::runtime_error("No more tensor in zip archive");
	}

	if (entries[tensor_order[next_tensor_index]].compression != 0)
	{
		throw std::runtime_error("Compression method " + std::to_string(entries[tensor_order[next_tensor_index]].compression) + " not supported");
	}

	return GetData(tensor_order[next_tensor_index++]);
}
