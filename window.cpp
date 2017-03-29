#include "window.h"

window::window(const std::string& name, unsigned int width, unsigned int height) {
	m_width = width;
	m_height = height;
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
	m_renderer = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_PRESENTVSYNC);
	if (nullptr == m_renderer)
		throw std::runtime_error("Could not create renderer.");
	m_texture = SDL_CreateTexture(m_renderer, PIXEL_FORMAT, SDL_TEXTUREACCESS_STREAMING, m_width, m_height);
	if (nullptr == m_texture)
		throw std::runtime_error("Could not create texture.");
}

window::~window() {
	SDL_DestroyTexture(m_texture);
	SDL_DestroyRenderer(m_renderer);
	SDL_DestroyWindow(m_window);
	SDL_Quit();
}

void window::show_window(std::vector<point_charge_t> charges) {
	cpu_computation solver;
	std::vector<uint32_t> result = solver.visualization(
			charges,
			m_width,
			m_height,
			PIXEL_FORMAT);
	if (SDL_UpdateTexture(m_texture, nullptr, &result[0], m_width * sizeof(uint32_t)))
		throw std::runtime_error(SDL_GetError());

	bool quit = false;
	SDL_Event e;
	while (!quit) {
		while (SDL_PollEvent(&e) != 0) {
			if (e.type == SDL_QUIT) {
				quit = true;
			}
		}
		SDL_RenderCopy(m_renderer, m_texture, nullptr, nullptr);
		SDL_RenderPresent(m_renderer);
	}
}
