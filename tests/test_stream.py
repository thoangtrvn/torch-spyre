import torch


def old_way():
    # Old way
    dev = torch.device("spyre")
    stream1 = torch.spyre.Stream(dev)
    print(stream1)
    print(torch.spyre.current_stream())
    stream2 = torch.spyre.Stream(priority=-1)  # High priority
    print(stream2)
    stream2 = torch.spyre.Stream(dev, priority=-1)  # High priority
    print(stream2)


def modern_way():
    # Modern way
    dev = torch.device("spyre")
    stream1 = torch.Stream(dev)
    print(stream1)

    stream2 = torch.Stream(dev, priority=-1)  # High priority
    print(stream2)

    # Use stream context
    with torch.Stream(dev):
        a = torch.randn(1, 32).to(device="spyre")
        print(a)

    # Query current stream
    current = torch.accelerator.current_stream()
    print(f"Current stream: {current}")

    # Synchronize
    stream1.synchronize()  # Wait for stream1
    torch.accelerator.synchronize(dev)  # Wait for all streams on given device
    torch.accelerator.synchronize()  # Wait for all streams


old_way()
modern_way()
