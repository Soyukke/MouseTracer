use bytemuck::{Pod, Zeroable};
use nanorand::{Rng, WyRand};
use std::{borrow::Cow, mem};
use enigo::Enigo;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window, dpi::{LogicalSize, LogicalPosition},
};
use image::io::Reader as ImageReader;
use wgpu::util::DeviceExt;

const WINDOW_WIDTH: f32  = 75.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    mvp: [[f32; 4]; 4],
    size: [f32; 2],
    pad: [f32; 2],
}

/**
 * 座標(x, y): [f32, 2]
 * 
 */
#[repr(C, align(256))]
#[derive(Clone, Copy, Zeroable)]
struct Locals {
    position: [f32; 2],
    velocity: [f32; 2],
    color: u32,
    _pad: u32,
}

fn get_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
        let texture = {
            let img_data = include_bytes!("../assets/image.png");
            let dyimg = image::load_from_memory(img_data).unwrap();
            //let decoder = png::Decoder::new(std::io::Cursor::new(img_data));
            //let mut reader = decoder.read_info().unwrap();
            //let mut buf = vec![0; reader.output_buffer_size()];
            //let info = reader.next_frame(&mut buf).unwrap();
            let dyimg2 = dyimg.into_rgba8();

            let size = wgpu::Extent3d {
                width: dyimg2.width(),
                height: dyimg2.height(),
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            });
            queue.write_texture(
                texture.as_image_copy(),
                &dyimg2.into_raw(),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: std::num::NonZeroU32::new(size.width * 4),
                    rows_per_image: None,
                },
                size,
            );
            texture
        };
        texture
}

// 参考
// https://zenn.dev/matcha_choco010/articles/2022-07-05-rust-graphics-wgpu

async fn run(event_loop: EventLoop<()>, window: Window) {

    let size = window.inner_size();
    println!("size: {:?}", size);
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    // Load the shaders from disk
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../res/shader.wgsl"))),
    });

    let global_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(mem::size_of::<Globals>() as _),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                ],
                label: None,
        });



    let local_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(mem::size_of::<Locals>() as _),
                },
                count: None,
            }],
            label: None,
        });

    let view = get_texture(&device, &queue).create_view(&wgpu::TextureViewDescriptor::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let swapchain_capabilities = surface.get_supported_formats(&adapter);
    let swapchain_format = swapchain_capabilities[0];
    println!("swapchain_format: {:?}", swapchain_format);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        // alpha_mode: swapchain_capabilities.alpha_modes[0],
        alpha_mode: wgpu::CompositeAlphaMode::PostMultiplied,
    };

    let globals = Globals {
        mvp: glam::Mat4::orthographic_rh(
                 0.0,
                 WINDOW_WIDTH,
                 0.0,
                 WINDOW_WIDTH,
                 -1.0,
                 1.0,
                 )
            .to_cols_array_2d(),
            size: [0.15*256.0; 2],
            pad: [0.0; 2],
    };

    let global_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("global"),
        contents: bytemuck::bytes_of(&globals),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
    });


    let uniform_alignment =
        device.limits().min_uniform_buffer_offset_alignment as wgpu::BufferAddress;
    let local_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("local"),
        size: (1 as wgpu::BufferAddress) * uniform_alignment,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });


    let global_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &global_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: global_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
        label: None,
    });


    let local_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &local_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &local_buffer,
                offset: 0,
                size: wgpu::BufferSize::new(mem::size_of::<Locals>() as _),
            }),
        }],
        label: None,
    });


    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&global_bind_group_layout, &local_bind_group_layout],
        push_constant_ranges: &[],
    });



    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(swapchain_format.into())],
        }),
        // 矩形の4点で描画する
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: Some(wgpu::IndexFormat::Uint16),
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    surface.configure(&device, &config);

    event_loop.run(move |event, _, control_flow| {


        // Mouseポジション
        let cursor_location: (i32, i32) = Enigo::mouse_location();
        //let p = LogicalPosition{x: cursor_location.0 - (WINDOW_WIDTH/2.0) as i32, y: cursor_location.1 - (WINDOW_WIDTH/2.0) as i32};
        let p = LogicalPosition{x: cursor_location.0, y: cursor_location.1};
        // println!("cursor: {:?}", cursor_location);
        // Windowポジションをマウスの場所へ変更
        window.set_outer_position(p);

        let _ = (&instance, &adapter, &shader, &pipeline_layout);

        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                // Reconfigure the surface with the new size
                // Window sizeにする
                surface.configure(&device, &config);
                // On macos the window needs to be redrawn manually after resizing
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let frame = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {

                    // localsに書き込む
                    let mut rng = WyRand::new_seed(42);
                    let color = rng.generate::<u32>();
                    let mut rects: Vec<Locals> = vec![];
                    let extent = [WINDOW_WIDTH, WINDOW_WIDTH];
                    rects.push(Locals {
                        position: [0.0, 0.5 * (extent[1] as f32)],
                        velocity: [0.0, 0.0],
                        color,
                        _pad: 0,
                    });
                    let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment;
                    queue.write_buffer(&local_buffer, 0, unsafe {
                        std::slice::from_raw_parts(
                            rects.as_ptr() as *const u8,
                            rects.len() * uniform_alignment as usize,
                            )
                    });
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                // load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &global_group, &[]);
                    let offset =
                        (0 as wgpu::DynamicOffset) * (uniform_alignment as wgpu::DynamicOffset);
                    rpass.set_bind_group(1, &local_group, &[offset]);
                    rpass.draw(0..4, 0..1);
                }

                queue.submit(Some(encoder.finish()));
                frame.present();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

fn main() {
    let event_loop = EventLoop::new();
    // let window = winit::window::Window::new(&event_loop).unwrap();
    let mut builder = winit::window::WindowBuilder::new()
                .with_decorations(false)
                .with_always_on_top(true)
                .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_WIDTH))
                .with_transparent(true);
    builder = builder.with_title("hello-triangle");
    #[cfg(windows_OFF)] // TODO
    {
        use winit::platform::windows::WindowBuilderExtWindows;
        builder = builder.with_no_redirection_bitmap(true);
    }
    let window = builder.build(&event_loop).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        // Temporarily avoid srgb formats for the swapchain on the web
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
