#pragma warning( disable : 4996 )
#include <stdio.h>
#include <CL/cl.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <chrono>

// constants definition
// simulating
const int gridWidth = 1600;
const int gridHeight = 800;
const int gridAmount = gridWidth * gridHeight;
const float velSquared = 8.0f;
const float cellSize = 0.015f;
const float deltaTime = 0.05f;
const float deltaTimeSquared = deltaTime * deltaTime;
const float cellSizeRev = 1 / cellSize;

// graphics
const int windowWidth = 1600;
const int windowHeight = 800;
const int unitWidth = windowWidth / gridWidth;
const int unitHeight = windowHeight / gridHeight;
const float colorRatio = 255.f;
const sf::Color backGroundColor = sf::Color::Black;

float* last = new float[gridAmount]();
float* grid = new float[gridAmount]();
float* next = new float[gridAmount]();

static void init(float x = 0.f)
{
    for (int i = 0; i < gridAmount; i++)
    {
        last[i] = x;
        grid[i] = x;
        next[i] = x;
    }
}

static void setValue(const int x, const int y, const float value)
{
    const int id = x + y * gridWidth;
    last[id] = value;
    grid[id] = value;
    next[id] = value;
}

static char* ReadKernelSourceFile(const char* filename, size_t* length) {
    FILE* file = NULL;
    size_t sourceLenth;
    char* sourceString;
    int ret;
    file = fopen(filename, "rb");
    if (file == NULL) {
        printf("%s at %d : Can't open %s \n", __FILE__, __LINE__ - 2, filename);
        return NULL;
    }
    //重定位到文件末尾
    fseek(file, 0, SEEK_END);
    sourceLenth = ftell(file);
    //重定位到文件开头
    fseek(file, 0, SEEK_SET);
    sourceString = (char*)malloc(sourceLenth + 1);
    sourceString[0] = '\0';
    ret = fread(sourceString, sourceLenth, 1, file);
    if (ret == 0) {
        printf("%s at %d : Cant't read source %s\n", __FILE__, __LINE__ - 2, filename);
        return NULL;
    }
    fclose(file);
    if (length != 0) {
        *length = sourceLenth;
    }
    sourceString[sourceLenth] = '\0';
    return sourceString;

}

int main()
{
    bool running = true;
    int counter = 0;
    init(0.f);
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "WaveSim3");
    
    // deal with opencl
    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform default_platform = platforms[0];
    std::vector<cl::Device> devices;
    err = default_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Failed to get devices" << std::endl;
        return 1;
    }
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context, device);
    size_t program_length;
    char* src = ReadKernelSourceFile("sim.cl", &program_length);
    cl::Program::Sources sources(1, std::make_pair(src, program_length));
    cl::Program program(context, sources);
    err = program.build({ device });
    if (err != CL_SUCCESS)
    {
        std::cerr << "Failed to build program" << std::endl;
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }
    cl::Kernel kernel(program, "iterate_gpu");
    std::vector<cl::Event> cl_events;
    cl::Buffer buffer_last(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * gridAmount, last);
    cl::Buffer buffer_grid(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * gridAmount, grid);
    cl::Buffer buffer_next(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * gridAmount, next);
    cl::Buffer buffer_color(context, CL_MEM_READ_WRITE, sizeof(cl_uchar4) * gridAmount);

    // deal with sfml
    sf::Texture texture;
    sf::Sprite sprite;
    sprite.setScale(unitWidth, unitHeight);
    texture.create(gridWidth, gridHeight);
    std::vector<cl_uchar4> colors(gridAmount);

    // simulation
    while (running)
    {
        auto clock_start = std::chrono::steady_clock::now();
        kernel.setArg(0, buffer_last);
        kernel.setArg(1, buffer_grid);
        kernel.setArg(2, buffer_next);
        kernel.setArg(3, gridWidth);
        kernel.setArg(4, velSquared * deltaTimeSquared);
        kernel.setArg(5, buffer_color);
        err = queue.enqueueNDRangeKernel(kernel, NULL, gridAmount, 200, &cl_events);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Failed to enqueue NDRange Kernel with error code " << err << std::endl;
            return 1;
        }
        
        
        err = queue.enqueueReadBuffer(buffer_color, CL_FALSE, 0, sizeof(cl_uchar4) * gridAmount, &colors[0], NULL);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to read color buffer with error code " << err << std::endl;
            return 1;
        }
        queue.enqueueReadBuffer(buffer_grid, CL_FALSE, 0, sizeof(float) * gridAmount, &last[0], NULL);
        queue.enqueueReadBuffer(buffer_next, CL_FALSE, 0, sizeof(float) * gridAmount, &grid[0], NULL);
        texture.update((const sf::Uint8*)&colors[0], gridWidth, gridHeight, 0, 0);
        queue.finish();
        sprite.setTexture(texture);
        window.clear(backGroundColor);
        window.draw(sprite);
        window.display();

        grid[100 + 200 * gridWidth] = 3.f * sin((float)counter * 0.08f);
        queue.enqueueWriteBuffer(buffer_last, CL_FALSE, 0, sizeof(float) * gridAmount, last);
        queue.enqueueWriteBuffer(buffer_grid, CL_FALSE, 0, sizeof(float) * gridAmount, grid);

        sf::Event sf_event;
        while (window.pollEvent(sf_event))
        {
            if (sf_event.type == sf::Event::Closed)
                window.close();
        }
        if (!window.isOpen())
        {
            running = false;
        }
        counter++;
        queue.finish();
        auto clock_end = std::chrono::steady_clock::now();
        auto duration = clock_end - clock_start;
        printf("frame: %d , time cost: %d ms\n", counter, std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
    }
}