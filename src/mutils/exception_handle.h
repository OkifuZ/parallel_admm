#pragma once

#include <stdexcept>
#include <iostream>
#include <string>

namespace ADU {
	
	inline void make_exception(const std::string& message, bool exit_=true) {
		try {
			throw std::runtime_error(message);
		}
		catch (const std::exception& e) {
			std::cerr << "Error: " << e.what() << std::endl;
			if (exit_) exit(-1);
		}
	}


}