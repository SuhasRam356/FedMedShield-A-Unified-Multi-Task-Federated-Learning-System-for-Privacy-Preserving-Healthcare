"""
═══════════════════════════════════════════════════════════════
FedMedShield — PostgreSQL Relational Models
Uses SQLAlchemy 2.0 async ORM for structured data:
- Users (Admins, Doctors, Researchers)
- Hospitals (Nodes in the FL network)
- FL Tasks (Sepsis, Tumor, etc.)
- Model Checkpoints (Global versions)
═══════════════════════════════════════════════════════════════
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Enum, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from .db_config import Base

class RoleEnum(str, enum.Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    DOCTOR = "doctor"

class TaskTypeEnum(str, enum.Enum):
    EHR = "ehr"
    IMAGING = "imaging"
    DRUG = "drug"
    IDS = "ids"

class User(Base):
    """System users (dashboard access)."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(RoleEnum), default=RoleEnum.RESEARCHER)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # A researcher can start multiple FL Tasks
    tasks = relationship("FLTask", back_populates="owner")


class Hospital(Base):
    """Federated Learning Nodes."""
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String(50), unique=True, index=True, nullable=False) # e.g., 'hospital-a'
    name = Column(String(100), nullable=False)
    region = Column(String(50))
    ip_address = Column(String(50))
    is_online = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_ping = Column(DateTime(timezone=True))


class FLTask(Base):
    """A specific Federated Learning orchestration task."""
    __tablename__ = "fl_tasks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    task_type = Column(Enum(TaskTypeEnum), nullable=False)
    status = Column(String(20), default="pending") # pending, running, completed, failed
    target_rounds = Column(Integer, default=5)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

    owner = relationship("User", back_populates="tasks")
    checkpoints = relationship("ModelCheckpoint", back_populates="task")


class ModelCheckpoint(Base):
    """Saves metadata about the global model versions (.npz files)."""
    __tablename__ = "model_checkpoints"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("fl_tasks.id"))
    round_number = Column(Integer, nullable=False)
    file_path = Column(String(255), nullable=False) # path to .npz file
    
    # Aggregated metrics for this round
    val_loss = Column(Float)
    val_metric = Column(Float) # e.g. Accuracy or F1
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    task = relationship("FLTask", back_populates="checkpoints")
