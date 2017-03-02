#include "window.h"

window::window(const std::string& name, unsigned int width, unsigned int height) {
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
		throw std::runtime_error("Could not initialize the SDL2 subsystem.");
	m_window = SDL_CreateWindow(
		name.c_str(),
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		width,
		height,
		SDL_WINDOW_SHOWN
	);
	if (nullptr == m_window)
		throw std::runtime_error("Could not create window.");
	m_surface = SDL_GetWindowSurface(m_window);
	if (nullptr == m_surface)
		throw std::runtime_error("Could not get the surface for the window.");
	SDL_FillRect(m_surface, nullptr, SDL_MapRGB(m_surface->format, 0xff, 0xff, 0xff));
	SDL_UpdateWindowSurface(m_window);
}

window::~window() {
	SDL_DestroyWindow(m_window);
	SDL_Quit();
}

void window::show_window() {
	bool quit = false;
	SDL_Event e;
	while (!quit) {
		while (SDL_PollEvent(&e) != 0) {
			if (e.type == SDL_QUIT) {
				quit = true;
			}
		}
	}
}
