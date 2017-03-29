#include "cpu_compute.h"
#include "gpu_compute.h"

#include "SDL2/SDL.h"
#include <string>
#include <stdexcept>

class window {
private:
	SDL_Window* m_window;
	SDL_Renderer* m_renderer;
	SDL_Texture* m_texture;

	unsigned int m_width, m_height;
	const Uint32 PIXEL_FORMAT = SDL_PIXELFORMAT_ARGB8888;
public:
	window(const std::string& name, unsigned int width, unsigned int height);
	~window();
	window(const window& window) = delete;
	window& operator=(const window& window) = delete;

	void show_window(std::vector<point_charge_t> charges);
};

