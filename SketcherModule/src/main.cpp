#include <iostream>
#include <string>
#include <thread>

#include "Canvas.h"

void StartFirstSession()
{
	Canvas::InitGLFWLib();

	std::thread top(Canvas::StartCanvas, Role::Top, 0);
	std::thread side(Canvas::StartCanvas, Role::Side, 0);
	std::thread persp(Canvas::StartCanvas, Role::Persp, 0);

	top.join();
	side.join();
	persp.join();
}

void StartNextIteration(size_t index)
{
	Canvas::InitGLFWLib();

	std::thread next(Canvas::StartCanvas, Role::Persp, index);
	next.join();
}

// To run in debug mode
int main()
{
	StartFirstSession();
}

// To use it as a dll
extern "C" {
	__declspec(dllexport) void StartSession()
	{
		StartFirstSession();
	}

	__declspec(dllexport) void StartNextSession(size_t index)
	{
		StartNextIteration(index);
	}
}