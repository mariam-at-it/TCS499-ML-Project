"""add user sessions

Revision ID: 0002_add_user_sessions
Revises: 0001_initial
Create Date: 2026-04-05 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_add_user_sessions"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("refresh_jti", sa.String(length=64), nullable=False),
        sa.Column("token_family", sa.String(length=64), nullable=False),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by_jti", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_user_sessions_refresh_jti", "user_sessions", ["refresh_jti"], unique=True)
    op.create_index("ix_user_sessions_token_family", "user_sessions", ["token_family"], unique=False)
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_user_sessions_user_id", table_name="user_sessions")
    op.drop_index("ix_user_sessions_token_family", table_name="user_sessions")
    op.drop_index("ix_user_sessions_refresh_jti", table_name="user_sessions")
    op.drop_table("user_sessions")
