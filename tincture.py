import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      Basic convenience functions:      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def clamp(value,minimum,maximum,):
    #choose the largest: value, or the minimum
    output = max(value, minimum)
    #then choose the smallest, of the max and the value
    output = min(value, maximum)
    return output

def wrap(value, minimum, maximum):
    """Basically a modulo wrapper, this convenience function wraps a value around,
    in the range of min to max."""
    output = value
    
    #translate the value by the min, because modulo works here on min=0
    output -= minimum
    deltamax = maximum - minimum
    
    #then modulo to wrap in our new range of 0 - deltamax. 
    output = output % deltamax
    
    #ok, now shift back to the original range
    output += minimum
    #lastly clamp so that we dont have any weird values due to rounding errors
    return clamp(output, minimum=minimum, maximum=maximum)

def blend_value(input1,input2,factor=0.5):
    """take two numbers, and average them to the weight of factor."""
    new_val = (input1 * (1 - factor)) + (input2 * factor)
    return new_val

def blend_tuple(input1,input2,factor=0.5):
    """take two tuples of equal size, and average them to the weight of factor.
    Mainly for blending colors."""
    output = []
    for i in range(0,len(input1)):
        val1 = input1[i]
        val2 = input2[i]
        new_val = (val1 * (1 - factor)) + (val2 * factor)
        output.append(new_val)
    return tuple(output)

def blend_angle(angle1, angle2, factor=0.5):
    """take two angles, and average them to the weight of factor.
    Mainly for blending hue angles."""
    # Ensure hue values are in the range [0, 360)
    angle1 = angle1 % 360
    angle2 = angle2 % 360

    # Calculate the angular distance between hue1 and hue2
    angular_distance = (angle2 - angle1 + 180) % 360 - 180

    # Calculate the middle hue value
    blended = (angle1 + angular_distance * factor) % 360

    return blended

def sigmoid(val):
    return 1 / (1 + np.exp(-val))

def ease_in_out(x:float) -> float:
    """easeInOutQuad function implementation"""
    if x < 0.5:
        return 2 * x**2
    else:
        return 1 - ((-2 * x + 2)**2) / 2
    
def map_range(value:float, input_min:float, input_max:float, output_min:float=0, output_max:float=1) -> float:
    """Take value, from range (input_min,input_max), and reshape it for the range (output_min,output_max)"""
    output = value - input_min
    output = output / (input_max - input_min)
    output = output * (output_max - output_min)
    output += output_min
    return output

def binary_search(sorted_list, target):
    """This helper function is used to quickly find the index of 2 numbers in a sorted list, 
    which are closest in value to the target value (one above and one below) """
    low = 0
    high = len(sorted_list) - 1

    # Handle cases where the target is outside the range of the list
    if target <= sorted_list[0]:
        return None, 0
    elif target >= sorted_list[-1]:
        return len(sorted_list) - 1, None

    while low <= high:
        mid = (low + high) // 2

        if sorted_list[mid] == target:
            # for consistency we should return mid twice, but if we do that, we'll waste time comparing them later.
            #so just return one, it should be a little faster. 
            return mid, None

        if sorted_list[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    # At this point, high is the index of the largest element < target,
    # and low is the index of the smallest element > target.

    return high, low

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      FUNCTIONS!      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def generate_pallete(num_colors: int, 
                     lightness_start: float = 20,lightness_end: float = 80, 
                     center_hue: float|None = None, hue_width: float|None = None,
                     saturation_start: float = 10, saturation_end: float = 70,
                     central_chroma_shift: float = 15, random_state: int|None = None,
                     debug: bool = False) -> list['Tinct']: 
    """Generate a color pallete randomly, or based on input parameters. 
    
    Returns a list of Tinct objects. 

    Input is based on HSL values, but pallete is transformed in okLCh.

    Parameters:
        - num_colors: how many colors in the final pallete
        - lightness_start: the beginning HSL lightness, in range 0 to 100
        - lightness_end: the final HSL lightness, in range 0 to 100
        - center_hue: central hue angle in degrees, or None for random
        - hue_width: distance between first and last hue angle in degrees, None for random
        - saturation_start: beginning saturation value for pallete, range 0 to 100
        - saturation_end: final saturation value for pallete, range 0 to 100
        - central_chroma_shift: percent to boost chromaticity in the middle range of pallete 
        - random_state: use a seed for the RNG, useful for keeping a good result
        - debug: print information about generation parameters
       
    """
    
    if num_colors < 1:
        raise ValueError(f'num_colors in generate_pallete was {num_colors}; it should be at least 1.')

    #init RNG 
    rng = np.random.default_rng(seed=random_state)

    #scale chroma shift to the useful range of [0,0.4] for LCh
    chroma_shift_val = central_chroma_shift * 0.004

    if hue_width == None:
        if num_colors > 10:
            hue_width = 180 * rng.normal()
        else:
            hue_width = (14 * num_colors) * rng.normal()
        hue_width = wrap(hue_width,-360,360)

        #hue_width = wrap(np.random.normal() * 180,-360,360)

    if center_hue == None:
        center_hue = rng.uniform(0,360)

    #start and end hues help us find our way
    start_hue = (center_hue - (hue_width * 0.5))
    end_hue = (center_hue + (hue_width * 0.5))

    #how much to incriment hue angle each step. This is passed in LCh back to the Tincts later.
    #without this, we cant have hue angles more than 180 (because the blend function takes the shortest path.)
    hue_inc = (end_hue - start_hue) / num_colors


    output = []
    start_color = Tinct(HSL=(start_hue, saturation_start, lightness_start))
    end_color = Tinct(HSL=(end_hue, saturation_end, lightness_end))

    if num_colors == 1:
        return [start_color.blend(end_color)]
    elif num_colors == 2:
        return [start_color,end_color]
    else:
        #increment over each requested color
        for i in range(0,num_colors):
            #fac is used to calculate blending between colors
            fac = i / (num_colors - 1)
            #this value increses in the middle, and is used to add chroma
            #mid = 1 - (abs(fac - 0.5) * 2)
            mid = 1 - abs((sigmoid((fac - 0.5) * 10) - 0.5) * 2)

            #call the blend method
            this_color = start_color.blend(end_color,factor=fac)
            l,c,h = this_color.get_okLCh()
            
            #add chroma for a more vibrant pallete. 
            c += chroma_shift_val * mid
            #increment our hue value. Doing it like this lets us have large hue angles.
            h = start_hue + (hue_inc * i)
            this_color.set_okLCh((l,c,h))
            
            output.append(this_color)

    if debug:
        #print some useful information about how the pallete was generated
        print('')
        print(f'Generate Pallete, num_colors = {num_colors}')
        print(f'Center hue = {center_hue}')
        print(f'Start hue = {start_hue}, End hue = {end_hue}, hue width = {hue_width}, hue increment = {hue_inc}')
        print(f'Start color: {start_color}, End color: {end_color}')

    return output





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      Tinct - color class definition      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Tinct:
    """'To tinge or tint / tinged; colored; flavored' - 
    This class aims to provide a very simple and straightforward way to mix, convert, and manipulate color.
    Colors are stored as a Tuple of floats in RGB, but can be converted an manipulated in any supported colorspace."""
    
    def __init__(self,RGB=None,RGBA=None,RGB255=None,HSL=None,okLab=None,okLCh=None,hex=None,position=None,alpha=1):
        self.rgb = (0.0,0.0,0.0)
        # I have chosen to store the value as an rgb value.
        # This is being done purely for convenience. Another colorspace might make more sense,
        # but this way it requires a minimum understanding of colorspaces to utilize the stored value.
        # However, the self.rgb value should always be raw; we need to be able to convert between
        # other colorspaces as well, even if those spaces are larger than RGB. 


        #check each of the input variables and preform required transformations.
        if RGB:
            self.set_RGB(RGB)
        elif RGBA:
            r,g,b,a = RGBA
            self.set_RGB((r,g,b))
            self.alpha = a
        elif RGB255:
            self.set_RGB255(RGB255)
        elif HSL:
            self.set_HSL(HSL)
        elif okLab:
            self.set_okLab(okLab)
        elif okLCh:
            self.set_okLCh(okLCh)
        elif hex:
            self.set_hex(hex)

        self.alpha = alpha

        #Tinct stores it's own position for use in Tincture
        self.position = position


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Useful modifiers:   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def blend(self,other_tinct:'Tinct',factor=0.5,ease=False) -> 'Tinct':
        """Mix this Tinct with another using factor. Transform is done is okLCh space. 
        if ease == True, apply an easing function to the factor (otherwise linear blend)"""
        if ease:
            fac = ease_in_out(factor)
        else:
            fac = factor
        l1,c1,h1 = self.get_okLCh()
        l2,c2,h2 = other_tinct.get_okLCh()
        l3,c3 = blend_tuple((l1,c1),(l2,c2),factor=fac)
        h3 = blend_angle(h1,h2,factor=fac)
        return Tinct(okLCh=(l3,c3,h3))
    
    # def gradient(self, other_tinct:'Tinct', num_steps:int) -> list['Tinct']:
    #     """Mix this Tinct with another, do so using a specified number of steps, and return as a list"""
    #     output = []
    #     if num_steps <= 2:
    #         raise ValueError(f'Number of gradient steps was {num_steps}, but it should be at least 3.')
    #     for i in range(0,num_steps):
    #         fac = (i / (num_steps - 1))
    #         output.append(self.blend(other_tinct,factor=fac))            
    #     return output

    def gradient(self, other_tinct:'Tinct',ease=False) -> 'Gradient':
        """Return a Gradient object that starts at this Tinct, and ends at other_tinct."""      
        return Gradient(RGB_start=self.get_RGB(), RGB_end=other_tinct.get_RGB(), ease=ease)
    
    def darker(self) -> 'Tinct':
        """Simply get a darker version of the color. Use add_lightness for a specific amount."""
        return self % 0
    
    def lighter(self) -> 'Tinct':
        """Simply get a lighter version of the color. Use add_lightness for a specific amount."""
        return self % 100
    
    def add_lightness(self,amount, clamped=True) -> 'Tinct':
        """Add amount (percent) to current lightness. Values can be negative.
        If clamped is True, resulting lightness will be clamped to range of 0-100%"""
        l,a,b = self.get_okLab()
        l = l + amount
        if clamped:
            l = clamp(l,0,100)
        return Tinct(okLab=(l,a,b))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   'set_' functions:   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_okLab(self,Lab):
        l,a,b = Lab
        #convert from percentages to 0,1 / -1,1 range
        #print(l,a,b)
        l *= 0.01
        a *= 0.01
        b *= 0.01
        #print(l,a,b)

        l_ = l + 0.3963377774 * a + 0.2158037573 * b
        m_ = l - 0.1055613458 * a - 0.0638541728 * b
        s_ = l - 0.0894841775 * a - 1.2914855480 * b

        l = l_*l_*l_
        m = m_*m_*m_
        s = s_*s_*s_

        rgb = self.linear_sRGB_transform((
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s))
        self.rgb = rgb

    def set_HSL(self,HSL):
        #convert HSL to RGB
        h,s,l = HSL
        #default should be percentage, so change to range 0 - 1
        s *= 0.01
        l *= 0.01
        chroma = (1 - abs((2 * l) - 1)) * s
        h_prime = h / 60
        X = chroma * (1 - abs(h_prime % 2 - 1))
        if 0 <= h_prime <= 1:
            tempRGB = (chroma,X,0)
        elif 1 <= h_prime <= 2:
            tempRGB = (X,chroma,0)
        elif 2 <= h_prime <= 3:
            tempRGB = (0,chroma,X)
        elif 3 <= h_prime <= 4:
            tempRGB = (0,X,chroma)
        elif 4 <= h_prime <= 5:
            tempRGB = (X,0,chroma)
        else:
            tempRGB = (chroma,0,X)
        m = l - (chroma / 2)
        r = tempRGB[0] + m; g = tempRGB[1] + m; b = tempRGB[2] + m
        self.rgb = (r,g,b)

    def set_RGB(self,RGB):
        """Input RGB as a 3 tuple of floats, range[0,1]"""
        self.rgb = RGB

    def set_RGBA(self,RGBA):
        """Input RGBA as a 4 tuple of floats, range[0,1]"""
        r,g,b,a = RGBA
        self.rgb = (r,g,b)
        self.alpha = a
    
    def set_RGB255(self,RGB255):
        """Input RGB as a 3 tuple of floats (or int) with range[0,255]"""
        r,g,b = RGB255
        self.rgb = (r/255,g/255,b/255)   
    
    def set_okLCh(self,LCh):
        """polar LCH to okLab"""
        l,chroma,hue = LCh

        a = chroma * np.cos(np.radians(hue))
        b = chroma * np.sin(np.radians(hue))
        self.set_okLab((l,a*100,b*100)) # oklab method expects a percentage input

    def set_hex(self,hex):
        """input an rgb hex"""
        cleaned = hex.replace('#','').strip()
        r,g,b = bytes.fromhex(cleaned)
        self.set_RGB255((r,g,b))

    def set_alpha(self,a):
        """update alpha value"""
        self.alpha = a

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 'get_' functions: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

    def get_RGB(self) -> tuple[float,float,float]:
        """return RGB value as tuple with range [0,1]"""
        #get the rgb color of this Tinct.
        r,g,b = self.rgb
        return (clamp(r,minimum=0,maximum=1),clamp(g,minimum=0,maximum=1),clamp(b,minimum=0,maximum=1))
    
    def get_RGBA(self) -> tuple[float,float,float,float]:
        """return RGBA value as a 4 tuple of floats, range [0,1]"""
        r,g,b = self.rgb
        return (clamp(r,0,1),clamp(g,0,1),clamp(b,0,1),clamp(self.alpha,0,1))


    def get_HSL(self,as_int=False) -> tuple[float,float,float]:
        """return HSL value as tuple with range:
        H [0,360], S [0,100%], L [0,100%]"""
        r,g,b = self.rgb
        Cmax = max(r,g,b)
        Cmin = min(r,g,b)
        delta = Cmax - Cmin

        #get hue
        if delta == 0:
            hue = 0
        elif Cmax == r:
            hue = 60 * (((g - b) / delta) % 6 )
        elif Cmax == g:
            hue = 60 * (((b-r+(2*delta)) / delta) % 6)
        else: #Cmax == b
            hue = 60 * (((r-g+(4*delta)) / delta) % 6)

        # get lightness
        lightness = (Cmax + Cmin) / 2

        #get saturation
        if delta == 0:
            saturation = 0
        else:
            saturation = delta / ( 1 - (abs( 2 * lightness - 1 )))

        #convert to percentage
        saturation *= 100
        lightness *= 100

        if as_int:
            hue = int(round(hue))
            saturation = int(round(saturation))
            lightness = int(round(lightness))

        return (hue,saturation,lightness)

    def get_RGB255(self, as_float=False) -> tuple[int,int,int]:
        """return okLab value as tuple with range:
        R [0,255], G [0,255], B [0,255]"""
        r,g,b = self.rgb
        r = clamp(r,0,255)
        g = clamp(g,0,255)
        b = clamp(b,0,255)
        if as_float:
            r = (r * 255)
            g = (g * 255)
            b = (b * 255)
        else:
            r = int(round(r * 255))
            g = int(round(g * 255))
            b = int(round(b * 255))
        return (r,g,b)
    
    def get_okLab(self) -> tuple[float,float,float]: 
        """return okLab value as tuple with range:
        L [0,100%], a [-100%,100%], b [-100%,100%]"""
        r,g,b = self.inverse_linear_sRGB_transform(self.rgb)
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

        l_ = np.cbrt(l)
        m_ = np.cbrt(m)
        s_ = np.cbrt(s)

        return (
        (0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_) * 100,
        (1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_) * 100,
        (0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_) * 100,
        )

    def get_okLCh(self) -> tuple[float,float,float]:
        """Converts value to L,C,h with ranges:
        L [0% : 100%], C [0 : 0.4ish], h [0 : 360]"""
        l,a,b = self.get_okLab()
        #convert values back to proper ranges
        a *= 0.01
        b *= 0.01

        chroma = np.sqrt((a ** 2) + (b ** 2)) #chroma
        hue = np.degrees(np.arctan2(b,a)) # hue degrees
        return (l,chroma,hue)

    def get_hex(self) -> str:
        """Get a hexidecimal representation of the color."""
        return '#%02x%02x%02x' % self.get_RGB255()

    def get_alpha(self) -> float:
        """Get this Tincts alpha, as a float range 0-1"""
        return clamp(self.alpha,0,1)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ dunders: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __len__(self):
        return 3 #len(self.rgb) should never change, so 3 is ok

    def __getitem__(self,index):
        rgb = self.get_RGB255()
        return rgb[index]

    def __str__(self):
        return self.get_hex()
    
    def __repr__(self):
        r,g,b = self.get_RGB255(as_float=True)
        return f'Tinct({r},{g},{b})'

    def __eq__(self, other):
        if type(other) == Tinct:
            return self.get_RGB255() == other.get_RGB255()
        else:
            return self.get_RGB255() == other

    def __int__(self):
        rgb = self.get_RGB255()
        result = ''
        for val in rgb:
            valstr = str(val)
            while len(valstr) < 3:
                valstr = '0' + valstr
            result += valstr
        return  int(result)
    
    # math

    def __add__(self,other):
        if type(other) in {int,float,bool}:
            r,g,b = self.rgb
            r += other
            g += other
            b += other
            return Tinct(RGB=(r,g,b))
        elif type(other) == Tinct:
            r,g,b = self.rgb
            r2,g2,b2 = other.rgb
            return Tinct(RGB=(r+r2,g+g2,b+b2))
        elif type(other) in {list,tuple}:
            if len(other) == 3:
                if type(other[0]) in {int,float,bool} and type(other[1]) in {int,float,bool} and type(other[2]) in {int,float,bool}:
                    r,g,b = self.rgb
                    r2,g2,b2 = other
                    return Tinct(RGB=(r+r2,g+g2,b+b2))
                else:
                    raise TypeError(f'Operations between Tinct and {type(other)} expect numerical values')
            else:
                raise TypeError(f'Operations between Tinct and {type(other)} expect that len({type(other)} == 3)')
        #if we made it here, its unsupported
        return NotImplemented
        
    def __sub__(self,other):
        if type(other) in {int,float,bool}:
            r,g,b = self.rgb
            r -= other
            g -= other
            b -= other
            return Tinct(RGB=(r,g,b))
        elif type(other) == Tinct:
            r,g,b = self.rgb
            r2,g2,b2 = other.rgb
            return Tinct(RGB=(r-r2,g-g2,b-b2))
        elif type(other) in {list,tuple}:
            if len(other) == 3:
                if type(other[0]) in {int,float,bool} and type(other[1]) in {int,float,bool} and type(other[2]) in {int,float,bool}:
                    r,g,b = self.rgb
                    r2,g2,b2 = other
                    return Tinct(RGB=(r-r2,g-g2,b-b2))
                else:
                    raise TypeError(f'Operations between Tinct and {type(other)} expect numerical values')
            else:
                raise TypeError(f'Operations between Tinct and {type(other)} expect that len({type(other)} == 3)')
        else:
            return NotImplemented
        
    def __mul__(self,other):
        if type(other) in {int,float,bool}:
            r,g,b = self.rgb
            r *= other
            g *= other
            b *= other
            return Tinct(RGB=(r,g,b))
        elif type(other) == Tinct:
            r,g,b = self.rgb
            r2,g2,b2 = other.rgb
            return Tinct(RGB=(r*r2,g*g2,b*b2))
        elif type(other) in {list,tuple}:
            if len(other) == 3:
                if type(other[0]) in {int,float,bool} and type(other[1]) in {int,float,bool} and type(other[2]) in {int,float,bool}:
                    r,g,b = self.rgb
                    r2,g2,b2 = other
                    return Tinct(RGB=(r*r2,g*g2,b*b2))
                else:
                    raise TypeError(f'Operations between Tinct and {type(other)} expect numerical values')
            else:
                raise TypeError(f'Operations between Tinct and {type(other)} expect that len({type(other)} == 3)')
        else:
            return NotImplemented
        
    def __truediv__(self,other):
        if type(other) in {int,float,bool}:
            r,g,b = self.rgb
            r /= other
            g /= other
            b /= other
            return Tinct(RGB=(r,g,b))
        elif type(other) == Tinct:
            # /0 operations are lazily avoided to allow for simple photo-editor-like behavior on Tinct/Tinct
            r,g,b = self.rgb
            r2,g2,b2 = other.rgb
            if r2 == 0:
                rr=r/0.00000000001
            else:
                rr = r / r2
            if g2 == 0:
                gg=g/0.00000000001
            else:
                gg = g / g2
            if b2 == 0:
                bb = b/0.00000000001
            else:
                bb = b / b2
            return Tinct(RGB=(rr,gg,bb))
        
        elif type(other) in {list,tuple}:
            if len(other) == 3:
                if type(other[0]) in {int,float,bool} and type(other[1]) in {int,float,bool} and type(other[2]) in {int,float,bool}:
                    r,g,b = self.rgb
                    r2,g2,b2 = other
                    return Tinct(RGB=(r/r2,g/g2,b/b2))
                else:
                    raise TypeError(f'Operations between Tinct and {type(other)} expect numerical values')

            else:
                raise TypeError(f'Operations between Tinct and {type(other)} expect that len({type(other)} == 3)')
        else:
            return NotImplemented
        
    def __mod__(self,other):
        #mod on tinct blends colors together
        if type(other) in {int,float,bool}:
            #if only one number is provided, just change lightness
            l,a,b = self.get_okLab()
            return self.blend(Tinct(okLab=(other,a,b)))
        elif type(other) == Tinct:
            # Tinct%Tinct shorthands a simple blend of colors
            return self.blend(other)
        elif type(other) in {list,tuple}:
            if len(other) == 3:
                if type(other[0]) in {int,float,bool} and type(other[1]) in {int,float,bool} and type(other[2]) in {int,float,bool}:
                    return self.blend(Tinct(RGB=(other)))
                else:
                    raise TypeError(f'Operations between Tinct and {type(other)} expect numerical values')
            else:
                raise TypeError(f'Operations between Tinct and {type(other)} expect that len({type(other)} == 3)')
        else:
            return NotImplemented
        
    def __radd__(self,other):
        if type(other) in {int,float,bool}:
            r,g,b = self.rgb
            r += other
            g += other
            b += other
            return Tinct(RGB=(r,g,b))
        elif type(other) in {list,tuple}:
            if len(other) == 3:
                if type(other[0]) in {int,float,bool} and type(other[1]) in {int,float,bool} and type(other[2]) in {int,float,bool}:
                    r,g,b = self.rgb
                    r2,g2,b2 = other
                    return Tinct(RGB=(r+r2,g+g2,b+b2))
                else:
                    raise TypeError(f'Operations between Tinct and {type(other)} expect numerical values')
            else:
                raise TypeError(f'Operations between Tinct and {type(other)} expect that len({type(other)} == 3)')
        #if we made it here, its unsupported
        return NotImplemented
        
    def __rsub__(self,other):
        if type(other) in {int,float,bool}:
            r,g,b = self.rgb
            r = other - r
            g = other - g
            b = other - b
            return Tinct(RGB=(r,g,b))
        elif type(other) in {list,tuple}:
            if len(other) == 3:
                if type(other[0]) in {int,float,bool} and type(other[1]) in {int,float,bool} and type(other[2]) in {int,float,bool}:
                    r,g,b = self.rgb
                    r2,g2,b2 = other
                    return Tinct(RGB=(r2-r,g2-g,b2-b))
                else:
                    raise TypeError(f'Operations between Tinct and {type(other)} expect numerical values')
            else:
                raise TypeError(f'Operations between Tinct and {type(other)} expect that len({type(other)} == 3)')
        else:
            return NotImplemented

    def __rmul__(self,other):
        if type(other) in {int,float,bool}:
            r,g,b = self.rgb
            r *= other
            g *= other
            b *= other
            return Tinct(RGB=(r,g,b))
        elif type(other) in {list,tuple}:
            if len(other) == 3:
                if type(other[0]) in {int,float,bool} and type(other[1]) in {int,float,bool} and type(other[2]) in {int,float,bool}:
                    r,g,b = self.rgb
                    r2,g2,b2 = other
                    return Tinct(RGB=(r*r2,g*g2,b*b2))
                else:
                    raise TypeError(f'Operations between Tinct and {type(other)} expect numerical values')
            else:
                raise TypeError(f'Operations between Tinct and {type(other)} expect that len({type(other)} == 3)')
        else:
            return NotImplemented
        
    def __rtruediv__(self,other):
        if type(other) in {int,float,bool}:
            r,g,b = self.rgb
            r = other / r
            g = other / g
            b = other / b
            return Tinct(RGB=(r,g,b))
        
        elif type(other) in {list,tuple}:
            if len(other) == 3:
                if type(other[0]) in {int,float,bool} and type(other[1]) in {int,float,bool} and type(other[2]) in {int,float,bool}:
                    r,g,b = self.rgb
                    r2,g2,b2 = other
                    return Tinct(RGB=(r2/r,g2/g,b2/b))
                else:
                    raise TypeError(f'Operations between Tinct and {type(other)} expect numerical values')

            else:
                raise TypeError(f'Operations between Tinct and {type(other)} expect that len({type(other)} == 3)')
        else:
            return NotImplemented

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ additional helper functions: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

    # linear srgb conversions are described here:
    # https://bottosson.github.io/posts/colorwrong/#what-can-we-do%3F
    def linear_sRGB_transform(self,sRGB:float) -> tuple[float,float,float]:
        output = []
        for x in sRGB:
            if x >= 0.0031308:
                output.append( (1.055 * x**(1.0/2.4)) - 0.055 )
            else:
                output.append( 12.92 * x )
        return tuple(output)
    
    def inverse_linear_sRGB_transform(self,RGB:float) -> tuple[float,float,float]:
        output = []
        for x in RGB:
            if x >= 0.04045:
                output.append( ((x + 0.055) / (1 + 0.055))**2.4 )
            else:
                output.append( x / 12.92 )
        return tuple(output)
    







#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    Gradient - color class definition     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Gradient:
    def __init__(self,
                 RGB_start=None,RGB255_start=None,HSL_start=None,okLab_start=None,okLCh_start=None,
                 RGB_end=None,RGB255_end=None,HSL_end=None,okLab_end=None,okLCh_end=None,
                 length = 100, ease=False):
        self.start = Tinct()
        self.end = Tinct(RGB=(1,1,1))
        self.length = length
        self.ease = ease

        #go through params to set our start and end colors
        if RGB_start:
            self.start.set_RGB(RGB_start)
        elif RGB255_start:
            self.start.set_RGB255(RGB255_start)
        elif HSL_start:
            self.start.set_HSL(HSL_start)
        elif okLab_start:
            self.start.set_okLab(okLab_start)
        elif okLCh_start:
            self.start.set_okLCh(okLCh_start)
        
        if RGB_end:
            self.end.set_RGB(RGB_end)
        elif RGB255_end:
            self.end.set_RGB255(RGB255_end)
        elif HSL_end:
            self.end.set_HSL(HSL_end)
        elif okLab_end:
            self.end.set_okLab(okLab_end)
        elif okLCh_end:
            self.end.set_okLCh(okLCh_end)
        
    def sample(self,fac:float) -> Tinct:
        return self.start.blend(self.end,factor=fac,ease=self.ease)
    
    def set_len(self,length:int):
        if length < 1:
            raise ValueError(f'Gradient was set to length of {length}, which is impossible.')
        self.length = length

    def __len__(self):
        return self.length
    
    def __getitem__(self,index):
        if self.length == 1:
            return self.start.blend(self.end,0.5)
        fac = 1 / (self.length - 1)
        return self.start.blend(self.end,fac)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    Tincture - color class definition     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
class Tincture:
    """To impart a tint or color to; To imbue or infuse with something.
    Params:
     - positions
     - blend_mode: 'linear' 'ease' or 'auto' to choose based on positions of content"""
    def __init__(self,*args,colors=None,positions=None, blend_mode='auto'):
        # args is the input colors. if input_colors is given, it overrides args. 
        # positions is 0-1, representing position of each Tinct.

        self.blend_mode=blend_mode

        input_colors = list(args)
        if len(input_colors) == 1:
            if type(input_colors[0]) in {list, tuple}:
                input_colors = input_colors[0]
        if colors:
            if type(colors) == Tinct:
                input_colors = [colors]
            else:
                input_colors = colors
        my_elements = []
        for item in input_colors:
            if type(item) == Tinct:
                my_elements.append(item)
            elif type(item) == Tincture:
                my_elements += item.elements
            elif type(item) == str:
                my_elements.append(Tinct(hex=item))
            elif type(item) in {list,tuple}:
                if len(item) == 3:
                    r,g,b = item
                    my_elements.append(Tinct(RGB=(r,g,b)))
                elif len(item) == 4:
                    r,g,b,a = item
                    my_elements.append(Tinct(RGBA=(r,g,b,a)))
                else:
                    raise TypeError(f'Tincture got a {type(item)} with len {len(item)} in input, but it should be a 3 tuple to interpret as RGB, or a 4 tuple for RGBA')
            else:
                raise TypeError(f'Tincture got a {type(item)} in input. Expected value of Tinct, Tincture, a 3-tuple representing RGB, or a 4-tuple representing RGBA')
            
        self.elements = my_elements
        if positions:
            self.update_positions(positions)
        else:
            self.set_default_positions()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    tincture utils    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update_positions(self,positions:list[float]):
        """Set positions for stored Tincts to the provided list of positions.
        Stored Tincts will also be sorted by position."""
        if len(positions) != len(self.elements):
            raise ValueError(f'Tincture contains {len(self.elements)} Tincts, but was given {len(positions)} positions.')
        for idx, pos in enumerate(positions):
            self.elements[idx].position = pos
        #sort ourselves
        self.sort_elements()

    def sort_elements(self):
        """Sort elements stored in this tincture. This is required for the retrieval functions to work properly."""
        self.elements = sorted(self.elements,key=lambda x: x.position)

    def set_default_positions(self):
        """Set the position values for stored Tincts to the default, which is a linear mapping of 0-1 based on position in list."""
        num_positions = len(self.elements)
        if num_positions == 1:
            #one position should just be 0, I suppose
            self.elements[0].position = 0.0
        elif num_positions > 0:
            for idx, tinct in enumerate(self.elements):
                fac = idx / (num_positions - 1)
                tinct.position = fac

    def reshape_positions(self):
        """Scale this Tincts position values so that they fit range 0-1."""
        positions = []
        for tinct in self.elements:
            positions.append(tinct.position)
        minimum = min(positions)
        maximum = max(positions)
        new_positions = []
        for pos in positions:
            new_positions.append(map_range(pos,minimum,maximum))
        self.update_positions(new_positions)

    def copy(self) -> 'Tincture':
        """return a copy of this Tincture"""
        return Tincture(colors=self.elements.copy(), blend_mode=self.blend_mode)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    modifying    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_tinct(self, tinct:Tinct = None, inplace:bool = False, **kwargs):
        """Quickly add Tinct to the Tincture. 
        Params:
         - tinct: pass a Tinct, to add that Tinct to this Tincture
         - inplace: False (default) will return a copy with your specified changes, True will return None and update this Tincture directly
         - **kwargs: instead of passing tinct, you can pass keyword args, which are simply passed on to a new Tinct object
        
        If the Tinct has no position, then it will be added to the end, and the other elements will be automatically reshaped."""
        my_tinct = None
        reshaping = False
        if tinct:
            if type(tinct) == Tinct:
                my_tinct = tinct
            else:
                raise TypeError(f"parameter 'tinct' expects a Tinct object, but got {type(tinct)}")
        if kwargs:
            my_tinct = Tinct(**kwargs)
        if my_tinct == None:
            raise TypeError("add_tinct needs to be given a Tinct to add (got None)")
        
        if my_tinct.position == None:
            reshaping = True
            #figure out where to put new element, based on previous elements location. This will be >1 to start, but fixed later.
            location = 1 + (1 / max(len(self.elements), 1))
            my_tinct.position = location
        

        if inplace:
            self.elements.append(my_tinct)
            if reshaping:
                self.reshape_positions()
            else:
                self.sort_elements()
            return None
        else:        
            new_elements = self.elements.copy()
            new_elements.append(my_tinct)
            #make a copy to do operations on
            new_tincture = self.copy()
            new_tincture.elements = new_elements
            if reshaping:
                new_tincture.reshape_positions()
            else:
                new_tincture.sort_elements()
            return new_tincture





    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    retrieval    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def sample(self, location):
        """Pick a color from the Tincture based on the specified position."""
        positions = []
        for tinct in self.elements:
            positions.append(tinct.position)

        # for starters, if the tincture has len 0, we can just return None
        if len(self.elements) == 0:
            return None
        #if len is 1, theres only one color to return, so do it
        elif len(self.elements) == 1:
            return self.elements[0]
        
        #if we passed both checks above, we have at least 2 elements, and we should select from them. 
        #use binary search to find where our location value falls compared to positions of the Tincts
        idx_low, idx_high = binary_search(positions, location)
        #if either values are empty, it means our location is outside of the range of self.elements[].position
        if idx_low == None:
            return self.elements[idx_high]
        elif idx_high == None:
            return self.elements[idx_low]
        #if neither check above passed, then we have two Tincts to sample from. 
        
        #lets figure out what positions our Tincts sit at
        pos_low = self.elements[idx_low].position
        pos_high = self.elements[idx_high].position
        #to use the built in blend function, we need to scale the position and location values from whatever they are now, to a range of 0-1

        #represents relative location between pos_low as 0, and pos_high as 1
        location_factor = map_range(location, pos_low, pos_high, 0, 1)

        #check our blend mode! 
        ease = (self.blend_mode == 'ease')
        if self.blend_mode == 'auto':
            #auto should use linear if we have len 3 or less.
            # but if we have more, then outer edges should be linear, others should be ease
            if len(self.elements) > 3 and (idx_low > 0 and idx_high < (len(self.elements) - 1)):
                ease = True

        #now lets simply sample from our range using the blend function, and our location factor
        return self.elements[idx_low].blend(self.elements[idx_high], location_factor, ease=ease)
    
    def get_list(self, num_colors, kind:None|str = None) -> list:
        """return a list of colors.
        params: 
         - num_colors: the length of the returned list
         - kind: 'RGB','RGBA','RGB255','HSL','okLab','okLCh','hex', or None.   
           - None will just return a list of Tincts. 
           - Setting a valid type will convert the Tinct's to color representations.
         """
        output = []
        if num_colors == 0:
            return []
        elif num_colors == len(self.elements):
            output = self.elements.copy()
        else:
            for i in range(0,num_colors):
                fac = i / (num_colors - 1)
                output.append(self.sample(fac))
        
        if type(kind):
            if kind.lower() == "rgb":
                tmp_output = []
                for tinct in output:
                    tmp_output.append(tinct.get_RGB())
                output = tmp_output
            elif kind.lower() == "rgba":
                tmp_output = []
                for tinct in output:
                    tmp_output.append(tinct.get_RGBA())
                output = tmp_output                
            
        return output




    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ dunders: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __len__(self):
        return len(self.elements)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Testing or demo body:   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    from PIL import Image, ImageDraw


    #pallete test:

    # pallete_len = 4
    # num_palletes = 4

    # nest_list = []
    # for i in range(0,num_palletes):
    #     gradient_list = generate_pallete(pallete_len,debug=True)
    #     colors_list = []
    #     # for tinct in gradient_list:
    #     #     colors_list.append(tinct.get_RGB255())
    #     nest_list.append(gradient_list)

    # img_x = 1920
    # img_y = 1080

    # output_image = Image.new("RGB", (img_x,img_y))

    # draw = ImageDraw.Draw(output_image)



    # draw.rectangle([(0,0),(img_x,img_y)],fill=(150,150,150))

    # x_pad = 32
    # y_pad = 32
    # pallete_width = img_x - x_pad * 2
   
    # bar_width = pallete_width / pallete_len
    # bar_height = img_y / num_palletes

    # for j in range(0,num_palletes):
        
    #     for i in range(0,pallete_len):

    #         x1 = bar_width * i
    #         x2 = x1 + bar_width
    #         x1 = int(clamp(x1, 0, img_x))
    #         x2 = int(clamp(x2, 0, img_x))

    #         y1 = bar_height * j
    #         y2 = y1 + bar_height

    #         draw.rectangle([(x1 + x_pad,y1 + y_pad),(x2 + x_pad,y2 - y_pad)], fill=tuple(nest_list[j][i]))

    
    # output_image.show()




    ###gradient test:
    
    img_x = 1024
    img_y = 256

    output_image = Image.new("RGB", (img_x,img_y))

    draw = ImageDraw.Draw(output_image)

    num_colors = 1024
    start = Tinct(RGB255=(0,16,32))
    middle = Tinct(RGB255=(155,16,170))
    middle2 = Tinct(RGB255=(155,255,255))
    end = Tinct(RGB255=(251,212,32))
    #end = start.add_lightness(1)
    tincture = Tincture([start, end], blend_mode='auto')


    #print(tincture.elements[0].position, tincture.elements[1].position)
    #print(len(tincture))

    #new_clr = tincture.sample(0.5)
    #tincture = tincture.add_tinct(RGB=(1,1,1)).add_tinct(RGB=(1,0,0))

    print('')
    print('')
    print(tincture.get_list(10,kind='RGBA'))



    #gradient = start.gradient(end,num_colors)
    #gradient = Gradient(RGB255_start=(155,16,170), RGB255_end=(251,212,32))
    #gradient = new_clr.gradient(new_clr,ease=False)

    # grad_clrs = []
    # for tinct in gradient:
    #     grad_clrs.append(tinct.get_RGB255())
    
    bar_width = img_x / num_colors
    for i in range(0,num_colors):
        fac = (i / (num_colors - 1))

        x1 = bar_width * i
        x2 = x1 + bar_width
        x1 = int(clamp(x1, 0,img_x))
        x2 = int(clamp(x2, 0,img_x))
        draw.rectangle([(x1,0),(x2,img_y)], fill=tincture.sample(fac).get_RGB255())

    
    output_image.show()


