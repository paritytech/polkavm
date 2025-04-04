// These match ASCII.
pub const ESCAPE: u8 = 0x1b;
pub const BACKSPACE: u8 = 0x08;

// These are picked arbitrarily.
pub const UPARROW: u8 = 0x80;
pub const DOWNARROW: u8 = 0x80 + 1;
pub const RIGHTARROW: u8 = 0x80 + 2;
pub const LEFTARROW: u8 = 0x80 + 3;

pub const F1: u8 = 0x80 + 4;
pub const F2: u8 = 0x80 + 5;
pub const F3: u8 = 0x80 + 6;
pub const F4: u8 = 0x80 + 7;
pub const F5: u8 = 0x80 + 8;
pub const F6: u8 = 0x80 + 9;
pub const F7: u8 = 0x80 + 10;
pub const F8: u8 = 0x80 + 11;
pub const F9: u8 = 0x80 + 12;
pub const F10: u8 = 0x80 + 13;
pub const F11: u8 = 0x80 + 14;
pub const F12: u8 = 0x80 + 15;

pub const CAPSLOCK: u8 = 0x80 + 16;

pub const PRTSCR: u8 = 0x80 + 17;
pub const SCRLCK: u8 = 0x80 + 18;
pub const PAUSE: u8 = 0x80 + 19;

pub const INS: u8 = 0x80 + 20;
pub const DEL: u8 = 0x80 + 21;
pub const HOME: u8 = 0x80 + 22;
pub const END: u8 = 0x80 + 23;
pub const PGUP: u8 = 0x80 + 24;
pub const PGDN: u8 = 0x80 + 25;

pub const LSHIFT: u8 = 0x80 + 26;
pub const RSHIFT: u8 = 0x80 + 27;
pub const LCTRL: u8 = 0x80 + 28;
pub const RCTRL: u8 = 0x80 + 29;
pub const LALT: u8 = 0x80 + 30;
pub const RALT: u8 = 0x80 + 31;

pub const MOUSE_1: u8 = 0x80 + 32;
pub const MOUSE_2: u8 = 0x80 + 33;
pub const MOUSE_3: u8 = 0x80 + 34;

pub const MOUSE_X: u8 = 0x80 + 35;
pub const MOUSE_Y: u8 = 0x80 + 36;

pub const MOUSE_WHEEL_UP: u8 = 0x80 + 37;
pub const MOUSE_WHEEL_DOWN: u8 = 0x80 + 38;

pub fn from_sdl2_wheel(dir: sdl2::mouse::MouseWheelDirection) -> Option<u8> {
    use sdl2::mouse::MouseWheelDirection as D;
    Some(match dir {
        D::Normal => MOUSE_WHEEL_DOWN,
        D::Flipped => MOUSE_WHEEL_UP,
        _ => return None,
    })
}

pub fn from_sdl2_mouse(button: sdl2::mouse::MouseButton) -> Option<u8> {
    use sdl2::mouse::MouseButton as M;
    Some(match button {
        M::Left => MOUSE_1,
        M::Right => MOUSE_2,
        M::Middle => MOUSE_3,
        _ => return None,
    })
}

pub fn from_sdl2_scancode(key: sdl2::keyboard::Scancode) -> Option<u8> {
    use sdl2::keyboard::Scancode as K;
    Some(match key {
        K::Right => RIGHTARROW,
        K::Left => LEFTARROW,
        K::Up => UPARROW,
        K::Down => DOWNARROW,
        K::Escape => ESCAPE,
        K::F1 => F1,
        K::F2 => F2,
        K::F3 => F3,
        K::F4 => F4,
        K::F5 => F5,
        K::F6 => F6,
        K::F7 => F7,
        K::F8 => F8,
        K::F9 => F9,
        K::F10 => F10,
        K::F11 => F11,
        K::F12 => F12,
        K::Backspace => BACKSPACE,
        K::Pause => PAUSE,
        K::CapsLock => CAPSLOCK,
        K::ScrollLock => SCRLCK,
        K::PrintScreen => PRTSCR,
        K::Home => HOME,
        K::End => END,
        K::PageUp => PGUP,
        K::PageDown => PGDN,
        K::Insert => INS,
        K::Delete => DEL,
        K::LCtrl => LCTRL,
        K::RCtrl => RCTRL,
        K::LShift => LSHIFT,
        K::RShift => RSHIFT,
        K::LAlt => LALT,
        K::RAlt => RALT,

        K::Return => b'\n',
        K::Tab => b'\t',

        K::Grave => b'`',
        K::Space => b' ',
        K::Equals => b'=',
        K::Minus => b'-',
        K::A => b'a',
        K::B => b'b',
        K::C => b'c',
        K::D => b'd',
        K::E => b'e',
        K::F => b'f',
        K::G => b'g',
        K::H => b'h',
        K::I => b'i',
        K::J => b'j',
        K::K => b'k',
        K::L => b'l',
        K::M => b'm',
        K::N => b'n',
        K::O => b'o',
        K::P => b'p',
        K::Q => b'q',
        K::R => b'r',
        K::S => b's',
        K::T => b't',
        K::U => b'u',
        K::V => b'v',
        K::W => b'w',
        K::X => b'x',
        K::Y => b'y',
        K::Z => b'z',
        K::Num0 => b'0',
        K::Num1 => b'1',
        K::Num2 => b'2',
        K::Num3 => b'3',
        K::Num4 => b'4',
        K::Num5 => b'5',
        K::Num6 => b'6',
        K::Num7 => b'7',
        K::Num8 => b'8',
        K::Num9 => b'9',

        K::Comma => b',',
        K::Period => b'.',
        K::Slash => b'/',
        K::Semicolon => b';',
        K::LeftBracket => b'[',
        K::Backslash => b'\\',
        K::RightBracket => b']',

        K::KpDivide => b'/',
        K::KpMultiply => b'*',
        K::KpMinus => b'-',
        K::KpPlus => b'+',
        K::KpEnter => b'\n',
        K::KpPeriod => b'.',
        K::KpEquals => b'=',

        K::Kp0 => b'.',
        K::Kp1 => END,
        K::Kp2 => DOWNARROW,
        K::Kp3 => PGDN,
        K::Kp4 => LEFTARROW,
        K::Kp5 => b'5',
        K::Kp6 => RIGHTARROW,
        K::Kp7 => HOME,
        K::Kp8 => UPARROW,
        K::Kp9 => PGUP,

        _ => return None,
    })
}
