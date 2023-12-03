#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#include "ShapeRecognizer.h"
#include "stb/stb_image_write.h"
#include "stb/stb_image.h"

#include <fstream>
#include <sstream>
#include <array>
#include <unordered_set>

enum class Role
{
    Top,
    Side,
    Persp
};

namespace {
    static GLuint CompileShader(GLenum type, const std::string& source)
    {
        GLuint id = glCreateShader(type);
        const char* src = source.c_str();
        glShaderSource(id, 1, &src, nullptr);
        glCompileShader(id);

        int result;
        glGetShaderiv(id, GL_COMPILE_STATUS, &result);
        if (result == GL_FALSE)
        {
            int length;
            glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
            char* message = (char*)alloca(length * sizeof(char));
            glGetShaderInfoLog(id, length, &length, message);
            std::cout << "Failed to compile shader!" << std::endl;
            std::cout << message << std::endl;
            glDeleteShader(id);
            return 0;
        }

        return id;
    }

    static GLuint CreateShaders(const std::string& vertexShader, const std::string& fragmentShader)
    {
        GLuint program = glCreateProgram();
        GLuint vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
        GLuint fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

        glAttachShader(program, vs);
        glAttachShader(program, fs);
        glLinkProgram(program);
        glValidateProgram(program);

        glDeleteShader(vs);
        glDeleteShader(fs);

        return program;
    }

    static std::string ParseShader(const std::string& filepath)
    {
        std::ifstream stream(filepath);

        std::string line;
        std::stringstream ss;
        while (getline(stream, line))
        {
            ss << line << '\n';
        }

        return ss.str();
    }
}

class Canvas
{
public:
    static void StartCanvas(Role role, size_t index = 0)
    {
        auto canvas = Canvas(role, index);
        canvas.ShowCanvas();
    }

    static bool InitGLFWLib()
    {
        /* Initialize the library */
        if (!glfwInit())
        {
            return false;
        }
    }

    Canvas(Role role, int index = 0) : _role(role), _index(index)
    {
    }

	
private:
    void ShowCanvas()
    {
        if (!InitGLFW())
        {
            std::cout << "Error in GLFW init" << std::endl;
            return;
        }

        InitShaders();
        InitBuffers();

        /* Loop until the user closes the window */
        while (!glfwWindowShouldClose(_window))
        {
            HandleMouseEvents();
            HandleKeyEvents();

            if (_needBufferUpdate)
            {
                UpdateBuffer(_vaoTmp, _pixelBufferTmp, _pixelsTmp);
                _needBufferUpdate = false;
            }
            if (_needPersisting)
            {
                if (!_pixelsTmp.empty())
                {
                    CommitNewShape();
                    UpdateBuffer(_vao, _pixelBuffer, _pixels);
                    _pixelsTmp.clear();
                }
                _needPersisting = false;
            }

            RenderDrawing();

            /* Swap front and back buffers */
            glfwSwapBuffers(_window);

            if (_saveImage)
            {
                SaveImage();
                _saveImage = false;
                break;
            }

            /* Poll for and process events */
            glfwPollEvents();
        }
    }

    bool InitGLFW()
    {
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        /* Create a windowed mode window and its OpenGL context */
        _window = glfwCreateWindow(_width, _height, GetOutpuFileName().data(), NULL, NULL);
        SetWindowPosition();
        if (!_window)
        {
            glfwTerminate();
            return false;
        }

        GLFWimage icons[1];
        icons[0].pixels = stbi_load("icon.png", &icons[0].width, &icons[0].height, 0, 4);
        glfwSetWindowIcon(_window, 1, icons);
        stbi_image_free(icons[0].pixels);

        /* Make the window's context current */
        glfwMakeContextCurrent(_window);

        if (!glewInit() == GLEW_OK)
        {
            return false;
        }

        return true;
    }

    void InitShaders()
    {
        // Shaders should be copied next to the executable with the relative path:
        std::string vertexSource = ParseShader("./shaders/vsSimpleQuad.shader");
        std::string fragmentSource = ParseShader("./shaders/fsTexturedQuad.shader");

        GLuint program = CreateShaders(vertexSource, fragmentSource);
        glUseProgram(program);

        _colorUniform = glGetUniformLocation(program, "drawColor");

        _shaderProgram = program;
    }

    void InitBuffers()
    {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glGenVertexArrays(1, &_vao);
        glGenBuffers(1, &_pixelBuffer);
        glBindVertexArray(_vao);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, _pixelBuffer);
        glVertexAttribPointer((GLuint)0, 2, GL_FLOAT, GL_FALSE, sizeof(DLA::Point), (GLvoid*)0);

        glGenVertexArrays(1, &_vaoTmp);
        glGenBuffers(1, &_pixelBufferTmp);
        glBindVertexArray(_vaoTmp);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, _pixelBufferTmp);
        glVertexAttribPointer((GLuint)0, 2, GL_FLOAT, GL_FALSE, sizeof(DLA::Point), (GLvoid*)0);
    
        glBindVertexArray(0);

        glPointSize(4.0f);
    }

    void RenderDrawing()
    {
        glClear(GL_COLOR_BUFFER_BIT);

        glBindVertexArray(_vao);
        glUniform3f(_colorUniform, 0.0, 0.0, 0.0);
        glDrawArrays(GL_POINTS, 0, _pixels.size());

        glBindVertexArray(_vaoTmp);
        glUniform3f(_colorUniform, 1.0, 0.0, 0.0);
        glDrawArrays(GL_POINTS, 0, _pixelsTmp.size());

        glBindVertexArray(0);
    }

    void HandleMouseEvents()
    {
        HandleButtonEvents();
        HandleCursorMoving();
    }

    void HandleCursorMoving()
    {
        if (_isDrawing)
        {
            double x, y;
            glfwGetCursorPos(_window, &x, &y);
            if (x > 0 && y > 0 && x < _width && y < _height)
            {
                DLA::Point newPixel{2.0 * ((float)x / _width - 0.5), 2.0 * (1.0 - (float)y / _height) - 1.0};

                if (std::find(_pixelsTmp.begin(), _pixelsTmp.end(), newPixel) == _pixelsTmp.end())
                {
                    _pixelsTmp.push_back(newPixel);
                    _needBufferUpdate = true;
                }
            }
        }
    }

    void HandleButtonEvents()
    {
        bool left = glfwGetMouseButton(_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        bool right = glfwGetMouseButton(_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

        _isDrawing = left || right;
        _needPersisting = !left && !right;
        _isEllipse |= right;
    }

    void HandleKeyEvents()
    {
        if (glfwGetKey(_window, GLFW_KEY_ENTER) == GLFW_PRESS)
        {
            _saveImage = true;
        }
    }

    void UpdateBuffer(GLuint vao, GLuint buffer, const std::vector<DLA::Point>& pixels)
    {
        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glBufferData(GL_ARRAY_BUFFER, pixels.size() * sizeof(DLA::Point), pixels.data(), GL_STATIC_DRAW);

        glBindVertexArray(0);
    }

    void CommitNewShape()
    {
        std::vector<DLA::Point> fittedPixels;
        if (_isEllipse)
        {
            Ellipse ellipse(_pixelsTmp);
            fittedPixels = ellipse.Tessellate(100);
        }
        else
        {
            Line line(_pixelsTmp);
            fittedPixels = line.Tessellate(100);
        }

        _pixels.insert(_pixels.end(), fittedPixels.begin(), fittedPixels.end());
        _isEllipse = false;
    }

    void SaveImage()
    {
        int width, height;
        glfwGetFramebufferSize(_window, &width, &height);

        GLsizei nrChannels = 3;
        GLsizei stride = nrChannels * width;
        stride += (stride % 4) ? (4 - stride % 4) : 0;
        GLsizei bufferSize = stride * height;
        std::vector<char> buffer(bufferSize);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glReadBuffer(GL_FRONT);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
        stbi_flip_vertically_on_write(true);
        stbi_write_png((GetOutpuFileName() + ".png").data(), width, height, nrChannels, buffer.data(), stride);
    }

    std::string GetOutpuFileName()
    {
        switch (_role)
        {
            case Role::Top: return "Top";
            case Role::Side: return "Side";
            case Role::Persp: return "Persp_" + to_string(_index);
        }
    }

    void SetWindowPosition()
    {
        // Arbitrary position, probably somewhere close to the center of the screen
        int x = 400;
        int y = 500;
        switch (_role)
        {
            case Role::Top: glfwSetWindowPos(_window, x, y); break;
            case Role::Side: glfwSetWindowPos(_window, x + _width + 4, y + _height + 42); break;
            case Role::Persp: glfwSetWindowPos(_window, x, y + _height + 42); break;
        }
    }

private:
    GLFWwindow* _window;
    Role _role;
    size_t _index;

    GLuint _shaderProgram;

    GLuint _vao;
    GLuint _vaoTmp;
    GLuint _pixelBuffer;
    GLuint _pixelBufferTmp;

    GLuint _colorUniform;

    std::vector<DLA::Point> _pixels;
    std::vector<DLA::Point> _pixelsTmp;

    bool _needBufferUpdate = false;
    bool _needPersisting = false;
    bool _isDrawing = false;
    bool _saveImage = false;
    bool _isEllipse = false;

    double _width = 256;
    double _height = _width;
};