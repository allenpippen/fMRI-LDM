from diffusers import PNDMScheduler
from diffusers import DiffusionPipeline

def get_scheduler():
    scheduler = PNDMScheduler.from_pretrained('./weights/diffsion_from_scratch.params', subfolder='scheduler')
    # scheduler = DiffusionPipeline.from_pretrained('../weights/diffsion_from_scratch.params', subfolder='scheduler')
    return scheduler

if __name__ == '__main__':
    scheduler = get_scheduler()
    print(scheduler)