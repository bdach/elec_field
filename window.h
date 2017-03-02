#include "SDL2/SDL.h"
#include <string>
#include <stdexcept>

class window {
private:
	SDL_Window* m_window;
	SDL_Surface* m_surface;
public:
	window(const std::string& name, unsigned int width, unsigned int height);
	~window();
	window(const window& window) = delete;
	window& operator=(const window& window) = delete;

	void show_window();
};

