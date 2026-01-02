use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Events that can be queued for processing
#[derive(Debug, Clone)]
pub enum Event {
    /// Create a new lifeform with initial cell
    CreateLifeform {
        lifeform_id: Option<usize>,
    },
    /// Add a new cell to an existing lifeform
    AddCellToLifeform {
        lifeform_id: usize,
    },
    /// Remove a cell from a lifeform
    RemoveCellFromLifeform {
        lifeform_id: usize,
    },
}

/// Thread-safe event queue for simulation events
pub struct EventQueue {
    sender: Sender<Event>,
    receiver: Receiver<Event>,
    pending_count: Arc<AtomicUsize>,
}

impl EventQueue {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            sender,
            receiver,
            pending_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Send an event to the queue
    pub fn send(&self, event: Event) -> Result<(), mpsc::SendError<Event>> {
        self.pending_count.fetch_add(1, Ordering::Relaxed);
        self.sender.send(event)
    }

    /// Try to receive an event without blocking
    pub fn try_recv(&self) -> Result<Event, mpsc::TryRecvError> {
        match self.receiver.try_recv() {
            Ok(event) => {
                self.pending_count.fetch_sub(1, Ordering::Relaxed);
                Ok(event)
            }
            Err(e) => Err(e),
        }
    }

    /// Get the number of pending events
    pub fn pending_count(&self) -> usize {
        self.pending_count.load(Ordering::Relaxed)
    }

    /// Check if there are any pending events
    pub fn has_pending(&self) -> bool {
        self.pending_count() > 0
    }

    /// Clear all pending events
    pub fn clear(&self) {
        while self.try_recv().is_ok() {}
    }
}
